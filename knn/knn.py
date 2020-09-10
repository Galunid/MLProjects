from csv import reader
from collections import Counter


def _csv_to_data(dataset):
    with open(dataset, 'r') as f:
        next(f)  # skip first line
        data = reader(f, delimiter=',')
        parsed_data = []
        for row in data:
            for index, val in enumerate(row):
                if index != len(row) - 1:
                    row[index] = float(val)  # convert strings to floats
            parsed_data.append(tuple(row))  # tuples instead of lists, because hashable element is needed in set()
    return list(set(parsed_data))  # strip duplicates


def _euclidean_distance(row1, row2):
    dist_squared = 0
    for col1, col2 in zip(row1[:-1], row2[:-1]):
        dist_squared += (col1 - col2)**2
    return dist_squared**0.5  # take root of distance


def knn(data, query, k, label_column=-1):
    neighbour_distances = []
    for index, row in enumerate(data):
        distance = _euclidean_distance(row, query)
        neighbour_distances.append((index, distance))
    sorted_distances = sorted(neighbour_distances, key=lambda x: x[1])
    k_nearest = sorted_distances[:k]
    k_nearest_labels = [data[i][label_column] for i, _ in k_nearest]
    counted = Counter(k_nearest_labels)
    return counted.most_common(1)[0][0]


def test():
    error = 0
    for idx, entry in enumerate(data):
        res = knn(data, entry, 9)
        if res != entry[4]:
            error += 1
            print(idx, res, entry[4])
    return error


if __name__ == '__main__':
    data = _csv_to_data('iris.csv')
    print(100 - (test() / len(data) * 100))
