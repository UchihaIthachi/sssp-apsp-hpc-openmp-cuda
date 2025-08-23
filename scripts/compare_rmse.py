import sys, math
def read_vec(path):
    vals=[]
    with open(path,'r') as f:
        for line in f:
            s=line.strip()
            if not s: continue
            if s.startswith("Negative weight cycle"):
                print(f"[warn] {path}: negative cycle; RMSE N/A"); sys.exit(0)
            if s=="INF": vals.append(float('inf'))
            else: vals.append(float(s))
    return vals
def rmse(a,b):
    assert len(a)==len(b)
    s=0.0; n=0
    for x,y in zip(a,b):
        if math.isinf(x) and math.isinf(y): continue
        s+=(x-y)**2; n+=1
    return math.sqrt(s/max(n,1))
if __name__=="__main__":
    if len(sys.argv)<3: print("Usage: python3 compare_rmse.py ref.txt test.txt"); sys.exit(1)
    a=read_vec(sys.argv[1]); b=read_vec(sys.argv[2])
    print(f"RMSE: {rmse(a,b):.6f}")
