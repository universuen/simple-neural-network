import myNetwork
mnw = myNetwork.NetWork([3, 2, 5], rate = 0.1)
mnw.backward([1,2,3,4,5], [1,2,3])
print(mnw.activate([1, 2, 3]))