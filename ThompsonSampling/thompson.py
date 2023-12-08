
import numpy as np

def main():
  print("Begin Thompson sampling demo ")
  print("Goal is to maximize payout from three machines")
  print("Machines pay out with probs 0.3, 0.7, 0.5")

  N = 3  # number machines
  means = np.array([0.3, 0.7, 0.5])
  probs = np.zeros(N)
  S = np.zeros(N, dtype=int)
  F = np.zeros(N, dtype=int)
  rnd = np.random.RandomState(7)

  for trial in range(10):
    print("\nTrial " + str(trial))

    for i in range(N): 
      probs[i] = rnd.beta(S[i] + 1, F[i] + 1)

    print("sampling probs =  ", end="")
    for i in range(N):
      print("%0.4f  " % probs[i], end="")
    print("")

    machine = np.argmax(probs)
    print("Playing machine " + str(machine), end="")

    p = rnd.random_sample()  # [0.0, 1.0)
    if p < means[machine]:
      print(" -- win")
      S[machine] += 1
    else:
      print(" -- lose")
      F[machine] += 1

  print("Final Success vector: ", end="")
  print(S)
  print("Final Failure vector: ", end="")
  print(F)

if __name__ == "__main__":
  main()