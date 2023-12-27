import random

from myNetwork import NetWork


point_network = NetWork([2, 16, 1], 0.01)


def generate_data(num_points):
    data = []
    for _ in range(num_points):
        x, y = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
        label = [1] if x**2 + y**2 <= 1 else [0]
        data.append(([x, y], label))
    return data


training_data = generate_data(1000)
test_data = generate_data(100)
correct_predictions = 0
for data, label in test_data:
    output = point_network.activate(data)
    predicted_label = 1 if output[0] >= 0.5 else 0
    if predicted_label == label[0]:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)

print(f"BEFORE TRAINING: Testing Point Classification Network:")
print(f"Accuracy: {accuracy * 100:.2f}%")


epochs = 100
for epoch in range(epochs):
    for data, label in training_data:
        output = point_network.activate(data)
        point_network.backward(label, data)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} done.")


correct_predictions = 0
for data, label in test_data:
    output = point_network.activate(data)
    predicted_label = 1 if output[0] >= 0.5 else 0
    if predicted_label == label[0]:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)

print(f"AFTER TRAINING: Testing Point Classification Network:")
print(f"Accuracy: {accuracy * 100:.2f}%")
