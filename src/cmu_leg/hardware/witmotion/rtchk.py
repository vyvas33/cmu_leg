"""
Realtime Update Test

This test logs timing data from WitMotion HWT906 IMU or Yost Labs 3-Space IMU v2.1 (TSS)
in a soft realtime loop and plots the loop timing performance.

The loop runs on an absolute-deadline schedule: clk advances by exactly dt every iteration
and slack is measured against it, so lateness is re-referenced each cycle instead of
accumulating. Note that the loop body only fetches imu.latest_data, so the work duration is
~1 us and these plots characterise the OS timer rather than any control computation.

Plot 1: Cycle Timing Budget
    x-axis is the ideal schedule clk (i*dt); y-axis is position within that cycle [ms].
    The dashed line is the deadline, dt = 1/RATE.

    - blue (onset): when the loop actually woke, measured from the ideal cycle start clk_i
    - red (end of calc): onset + work, i.e. when the iteration finished
    - red - blue is the work duration; dt - red is the slack left in the cycle

    Ideally both traces sit well below the dashed line.
    - blue above it: the OS scheduler/timer alone overruns the budget. Optimizing the
      loop body cannot help; fix with an RT policy, core pinning, or a busy-wait tail.
    - red above it with blue low: the work itself is too slow for the budget.
    - both above: scheduler overrun with compute stacked on top.

    The y-axis floor is clamped at 0, so a cycle that woke early is clipped from view.

Plot 2: Onset Jitter
    Distribution of the blue trace, log count. While deadlines are being met the sleep
    cancels out of the algebra and onset_i+1 is exactly the sleep overshoot of cycle i, so
    this is a histogram of how much time.sleep() oversleeps on this platform. After a miss
    it instead carries the previous overrun (onset_i+1 = -slack_i).

    Read the tail, not the mode: the max onset is the real design number for the loop rate.

Plot 3: Slack
    Distribution of dt - onset - work, log count. Samples at or below zero are missed
    deadlines, and their count is the miss rate. Since the work is ~0 here, this is close to
    Plot 2 mirrored about dt rather than an independent measurement.

    Slack is not idle CPU: time.sleep() releases the GIL, so this window is also the budget
    in which the driver's parsing thread drains the serial buffer and assembles frames.

Plot 4: IMU -> Host Data Latency
    host_time_s - time_s over the run. Both come from the driver, and neither is a raw
    timestamp, so read this as relative drift rather than transport delay:

    - host_time_s is an alpha-filtered arrival estimate (alpha = 0.02, so a time constant of
      ~50 frames). Per-packet jitter is smoothed away; only excursions slower than ~50 ms
      survive.
    - host_time_s is anchored to time_s at the first complete frame, which forces the first
      sample to 0 by construction. Absolute wire/parse latency is not observable this way.
    - time_s comes from the 0x50 packet, whose millisecond field is a whole-ms integer. At
      RATE = 1000 that quantisation step equals one sample period.

    Expect a linear ramp (sensor RTC vs host crystal drift, slope in ms/s = ppm offset) on a
    1 ms staircase. The loop also samples latest_data with zero-order hold and no freshness
    check, so whenever frames arrive slower than the loop ticks, repeated rows appear as flat
    treads in the trace.

Plot 5: Latency Distribution
    Histogram of Plot 4. Only meaningful if that trace is flat: if a drift ramp is present
    this measures the ramp's range, not a jitter distribution, and its width means nothing.

The console also prints the total advance of both clocks against DURATION, which is the
cleanest measurement here -- the ratio gives the sensor-vs-host clock offset directly.
"""

import gc
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cmu_leg.config.params import IMU

# pin to core 1 for realtime control
if sys.platform.startswith("linux"):
    from cmu_leg.utility.rtutils import pin_to_core

    pin_to_core(cpu_core=1)

gc.disable()

IMU_TYPE = "wit"  # "wit" (WitMotion HWT906) or "tss" (Yost Labs 3-Space)
RATE, LPF = 1000, 256
DURATION =60.0  # test time in [s]


# --- INIT --- #


if IMU_TYPE == "wit":
    from cmu_leg.hardware.witmotion.hwt906 import WitMotionSensor as ImuSensor
    from cmu_leg.hardware.witmotion.hwt906 import discover_port, discover_baud

else:
    print("IMU type unknown. Use either 'wit' or 'tss'")
    exit(1)

port = IMU.foot_port
baud = discover_baud(port)
imu = ImuSensor(port=port, baud_bps=baud)


# --- RUN SOFT REALTIME LOG LOOP --- #


try:
    imu.connect()
    imu.set_streaming_rate(RATE)
    imu.set_lpf(LPF)
    imu.set_outputs(ts=True, acc=True, gyro=True, mag=False, ang=False, quat=False)

    imu.show_config()

    clk = 0.0
    dt = 1 / RATE
    update_timer = 0.0
    update_timer_intvl = 0.1 * DURATION

    timing_log = np.nan * np.zeros((int(RATE * DURATION) + 100, 5), dtype=float)
    lix = 0

    # start streaming and wait for IMUs' parsers to be up and running
    imu.stream()
    while not imu.drain_data_buffer():
        time.sleep(0.1)

    # realtime loop
    start_time = time.perf_counter()

    while True:
        try:
            elapsed = time.perf_counter() - start_time

            data = imu.latest_data

            if elapsed >= update_timer:
                print(f"Elapsed: {elapsed:.1f}s / {DURATION:.1f}s")
                update_timer += update_timer_intvl

            if elapsed >= DURATION:
                break

            # soft realtime loop
            slack = dt - (time.perf_counter() - start_time - clk)
            timing_log[lix] = (clk, elapsed - clk, slack, data["host_time_s"], data["time_s"])
            lix += 1
            clk += dt

            time.sleep(max(0, slack))   

        except KeyboardInterrupt:
            print(" > CTRL-C by user. Terminating")
            break
finally:
    if imu is not None:
        imu.disconnect(report=True)


# --- SHOW TIMING RESULTS --- #

print(f"\n{lix} frames collected.\n")

data = timing_log[:lix]
clk = data[:, 0]
onset = data[:, 1]
slack = data[:, 2]
host_ts = data[:, 3]
imu_ts = data[:, 4]

print(f"imu_ts  total advance: {(imu_ts[-1] - imu_ts[0]):.3f} s  (expected {DURATION:.1f} s)")
print(f"host_ts total advance: {(host_ts[-1] - host_ts[0]):.3f} s")


dt_ms = 1e3 / RATE

onset_ms = onset * 1e3  # ms — loop onset jitter
slack_ms = slack * 1e3  # ms — remaining headroom after work
end_ms = dt_ms - slack_ms  # ms — end of work within cycle
latency_ms = (host_ts - imu_ts) * 1e3  # ms — host vs sensor clock offset

fig = make_subplots(
    rows=2,
    cols=3,
    column_widths=[0.7, 0.15, 0.15],
    subplot_titles=(
        "<b>Cycle Timing Budget</b>",
        "<b>Onset Jitter</b>",
        "<b>Slack</b>",
        "<b>IMU -> Host Data Latency</b>",
        "<b>Latency Distribution</b>",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.06,
)

blue = "#4C72B0"
red = "#E15759"
orange = "#F58518"
dashed_line = dict(line_dash="dash", line_color="gray", line_width=1)

# --- Row 1: Cycle Timing Budget --- #

#  Col 1: Timing budget
fig.add_trace(
    row=1,
    col=1,
    trace=go.Scattergl(
        x=clk,
        y=end_ms,
        name="end of calc",
        mode="lines",
        line=dict(width=2.0, color=red),
        legendgroup="budget",
        showlegend=True,
    ),
)
fig.add_trace(
    row=1,
    col=1,
    trace=go.Scattergl(
        x=clk,
        y=onset_ms,
        name="onset",
        mode="lines",
        line=dict(width=2.0, color=blue),
        legendgroup="budget",
        showlegend=True,
    ),
)
fig.add_hline(row=1, col=1, y=dt_ms, **dashed_line)
fig.update_yaxes(row=1, col=1, title_text="clk cycle (ms)", range=[0, max(dt_ms, end_ms.max())])
fig.update_xaxes(row=1, col=1, title_text="clk (s)")

# Col 2: Onset jitter histogram
fig.add_trace(row=1, col=2, trace=go.Histogram(x=onset_ms, nbinsx=50, showlegend=False, marker_color=blue))
fig.update_yaxes(row=1, col=2, type="log", title_text="count")
fig.update_xaxes(row=1, col=2, title_text="onset (ms)")

# Col 3: Slack histogram
fig.add_trace(row=1, col=3, trace=go.Histogram(x=slack_ms, nbinsx=50, showlegend=False, marker_color=red))
fig.update_yaxes(row=1, col=3, type="log", title_text="count")
fig.update_xaxes(row=1, col=3, title_text="slack (ms)")

# --- Row 2: Hardware Synchronization ---

# Col 1: Host-sensor latency
fig.add_trace(
    row=2,
    col=1,
    trace=go.Scattergl(
        x=clk, y=latency_ms, mode="lines", line=dict(width=1.5, color=orange), name="latency (host−imu)", showlegend=False
    ),
)
fig.update_yaxes(row=2, col=1, title_text="latency (ms)")
fig.update_xaxes(row=2, col=1, title_text="clk (s)")

# Col 2: Latency distribution
fig.add_trace(
    row=2,
    col=2,
    trace=go.Histogram(x=latency_ms, nbinsx=50, showlegend=False, marker_color=orange),
)
fig.update_yaxes(row=2, col=2, title_text="count")
fig.update_xaxes(row=2, col=2, title_text="latency (ms)")

fig.update_layout(
    title=dict(
        text=f"<b>Realtime Loop Performance Test  (Rate: {RATE}Hz, Duration: {DURATION}s)</b>",
        font=dict(size=17),
        pad=dict(b=20),
    ),
    height=1000,
    margin=dict(t=100),
    font=dict(size=13),
    legend=dict(
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="lightgray",
        borderwidth=3,
        font=dict(size=13),
    ),
)
fig.update_annotations(font=dict(size=13), yshift=10)  # subplot titles
fig.update_xaxes(ticklabelstandoff=5)
fig.update_yaxes(ticklabelstandoff=5)

fig.write_html(
    "output/rtchk.html", include_plotlyjs="cdn", full_html=True, config={"scrollZoom": True, "displayModeBar": True}
)
print("Plot saved as output/rtchk.html")
