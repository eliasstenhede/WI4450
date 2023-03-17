from sys import argv

if __name__=="__main__":
    mflops_per_sec = 0
    mbyte_per_sec = 0
    with open(argv[1]) as f:
        for line in f.read().split("\n"):
            if line.startswith("MFlops/s"):
                mflops_per_sec = float(line.split(":")[1].strip())
            if line.startswith("MByte/s"):
                mbyte_per_sec = float(line.split(":")[1].strip())
    
    gflop_per_sec = mflops_per_sec/1024
    gbyte_per_sec = mbyte_per_sec/1024
    intensity = mflops_per_sec/mbyte_per_sec
    print(f"intensity: {intensity:.3f}\nGflop/s:   {gflop_per_sec:.3f}\nGbyte/s:   {gbyte_per_sec:.3f}")


            
