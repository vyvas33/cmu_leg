from multiprocessing import Process, cpu_count
import time 

def counter(num):
    count = 0
    while count < num:
        count += 1

def main():
    t0 = time.perf_counter()
    a = Process(target=counter, args=(1000000000/2,))
    b = Process(target=counter, args=(1000000000/2,))
    # a.start()
    # b.start()

    print("CPU Count: ", cpu_count())

    print("finished in: ", time.perf_counter() - t0, "seconds")


if __name__ == "__main__":
    main()
