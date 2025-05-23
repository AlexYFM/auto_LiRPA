## First Demo: 

comparing head on active (with decision logic) airplanes with slightly different initial sets

Initial sets: 
(More branching)
```
[[-2, -5, -2, np.pi, np.pi/12, 100], [-1,5,-1, np.pi, np.pi/12, 100]] (All ownships)
[[-1001, -5, 249, 0,0, 100], [-999, 5, 250, 0,0, 100]]
```
(Less branching)
```
[[-2, 5, -2, np.pi, np.pi/12, 100], [-1,15,-1, np.pi, np.pi/12, 100]]
[[-1001, -15, 249, 0,0, 100], [-999, -5, 250, 0,0, 100]]
```

Config: 
X: 14, Y: 13, Z: 15
Time horizon:
10.0 

Cached runs:
demo1_guam_ver.txt
demo1_guam_sim.txt (only use if live isn’t working)
demo1_guam_ver_no_branching.txt
demo1_guam_sim_no_branching.txt

## Second Demo:
Second demo: comparing advisories with a single intruder and two intruders, one with a slightly higher altitude 

Initial sets: 
```
[[-10, -1010, -1, np.pi/2, np.pi/6, 100], [10, -990, 1, np.pi/2, np.pi/6, 100]]
[[1199, -1, 649, np.pi,0, 100], [1201, 1, 651, np.pi,0, 100]] (NPC)
        
[[-2001, 299, 849, 0,0, 100], [-1999, 301, 851, 0,0, 100]] (NPC, for second part of demo)
```

Config: Same dimensions as before
Time horizon: 20 

Cached runs:
demo2_guam_ver_original.txt
demo2_guam_ver_single_original.txt
demo2_guam_sim_original.txt
demo2_guam_sim_single_original.txt

## Third Demo:
Four planes 

Initial sets: 
```
[[-10, -1010, -1, np.pi/2, np.pi/6, 100], [10, -990, 1, np.pi/2, np.pi/6, 100]] (Car)
[[1199, -1, 649, np.pi,0, 100], [1201, 1, 651, np.pi,0, 100]] (Car)
[[-2001, 299, 849, 0,0, 100], [-1999, 301, 851, 0,0, 100]] (NPC)
[[-10, 990, -1, -np.pi/2, np.pi/6, 100], [10, 1010, 1, -np.pi/2, np.pi/6, 100]] (Car)
```

Cached runs:
demo3_guam_sim.txt