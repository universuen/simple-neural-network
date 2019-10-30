import myNetwork

mnw = myNetwork.NetWork([3, 2, 5], rate=0.1)
# 对于同一组数据反复学习
cnt = 0
keys = [1, 0, 0, 0, 0]
print("期望的输出:", keys)
while cnt <= 1000:
    mnw.backward(keys, [1, 2, 3])
    if not cnt % 100:
        print(cnt, "次训练后输出层激活值:", mnw.activate([1, 2, 3]))
    cnt = cnt + 1
