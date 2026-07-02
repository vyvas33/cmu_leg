# Standing Tests

Earlier standing tests failed. Notably, there were oscillations in both the knee and the ankle motors. First suspect was the control frequency, and the time that is being used to compute the commands. This could be an issue in some sense, but not really the reason for observed oscillations. 

Further, probing into the main loop revealed a few problems. For context, when read_encoders was run, a bias of ~5Nm in torque was observed. This was ignored since it was assumed to be the moment due to the leg's own weight. (or the ground). However, due to this bias, even when a zero torque is desired by the high level controller, the low level controller outputs a non zero velocity and torque command which poisons the standing trigger. Hence, the oscillations start well before standing is set to true and are persistent throughout. Directly commanding a zero velocity instead of relying on the high level controller solved the issue of oscillations. Only the knee worked in these tests though.

The ankle has a faulty joint encoder, which is most likely a cabling issue (or solering), which has to be solved before conduction more tests. 
