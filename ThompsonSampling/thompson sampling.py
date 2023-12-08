import numpy as np

def main():
    print("Begin Thompson Sampling demo")
    print("Goal is to maximize payout from three machines")
    print("machines pay out with probabilities 0.3,0.7,0.5")

    N=3
    means=np.array([0.3,0.7,0.5])
    probs=np.zeros(N)
    S=np.zeros(N, dtype=int)
    F=np.zeros(N, dtype=int)
    rnd=np.random.RandomState(7)

    for trial in range(10):
        print("\nTrial " + str(trial))

        for i in range(N):
            probs[i]=rnd.beta(S[i]+1,F[i]+1)
        
        print("sampling probs= ", end="")
        for i in range(N):
            print("%0.4f " % probs[i], end="")
        print("")

        machine=np.argmax(probs)
        print("Playing Machine "+ str(machine), end="")

        p=rnd.random_sample()
        if p<means[machine]:
            print("SUCESS")
            S[machine]+=1
        else:
            print("FAIL")
            F[machine]+=1
        
    print("final sucess vector is: ")
    print(S)
    print("final failure vector is: ")
    print(F)

if __name__=="__main__":
    main()