import csv

import cv2
import operator
import numpy as np
from scipy.spatial import distance
import time
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from scipy.linalg import hadamard
import itertools


# to convert number to bit array 8 -> [0, 0, 0, 0, 1, 0, 0, 0]
def bitfield(n):
    result = [int(digit) for digit in bin(n)[2:]]
    len_result = len(result)
    new_result = []
    if len_result < 8:
        amount_of_zeros = 8 - len_result
        for i in range(amount_of_zeros):
            new_result.append(0)
        for i in range(len(result)):
            new_result.append(result[i])
        return new_result
    return result


# to convert 32-byte format descriptor to 256-bit format
def return_array_of_256_bits(point):
    result_array = []
    for byte in range(len(point)):
        bit_point = bitfield(point[byte])
        for x in range(len(bit_point)):
            result_array.append(bit_point[x])
    return result_array


# to convert all 32-byte format descriptors to 256-bit format
def convert_32_descriptors_to_256_bit(descriptors):
    result_matrix = []
    for keypoint in range(len(descriptors)):
        a = return_array_of_256_bits(descriptors[keypoint])
        result_matrix.append(a)
    return result_matrix


# convert bit representation of descriptor to integer.
# it is enough to multiply each descriptor, which consists of a vector of 256 bit elements long, by the Hadamard matrix.
# [0, 1, 1, 1, 0, 1, 1, ..., 1, 0] --> [132, 4, 14, -2, -8, -4, ..., 2, 2]
def convert_bit_format_descriptors_to_integers_format(descriptors_bit_format):

    # https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%90%D0%B4%D0%B0%D0%BC%D0%B0%D1%80%D0%B0
    adamar_matrix = hadamard(256)
    converted_descriptors = []
    for i in range(len(descriptors_bit_format)):
        # print(i)
        converted_descriptors.append(converted_descriptor_by_adamar_matrix(descriptors_bit_format[i], adamar_matrix))
    return converted_descriptors


# multiplication of the vector representation of the descriptor by the Hadamard matrix
def converted_descriptor_by_adamar_matrix(descriptor, adamar_matrix):
    result_array = []
    # for i in range(len(descriptor)):
    #     sum = 0
    #     for j in range(len(descriptor)):
    #         sum = sum + (descriptor[j] * adamar_matrix[i][j])
    #     result_array.append(sum)
    # return result_array

    result_array = np.dot(descriptor, adamar_matrix)
    return result_array


# def get_dispersion_for_etalon(etalon, index):
#     sum_for_average = 0
#     # average = 0
#
#     for j in range(len(etalon)): # range 1000
#         sum_for_average = sum_for_average + etalon[j][index]
#     average = sum_for_average / len(etalon)
#
#     sum = 0
#     for j in range(len(etalon)):
#          sum = sum + ((etalon[j][index] - average) ** 2)
#
#     res = sum / (len(etalon) - 1)
#     return round(res, 2)


# to calculate dispersion for each etalon's column
def get_dispersion_for_etalon(etalons_descriptors, index, degree):

    # each column of the etalon is taken.
    sum_for_average = 0
    average = 0

    # the average value of the column is calculated.
    for j in range(len(etalons_descriptors)): # range 1000
        sum_for_average = sum_for_average + etalons_descriptors[j][index]
    average = sum_for_average / len(etalons_descriptors)

    sum = 0

    #   The average value is subtracted from each element of the column.
    #   The difference is squared. Each difference for each element is summed.
    if degree:
        for j in range(len(etalons_descriptors)):
             sum = sum + (((etalons_descriptors[j][index] ** 2) - (average ** 2)) ** 2)
    else:
        for j in range(len(etalons_descriptors)):
             sum = sum + ((etalons_descriptors[j][index] - average) ** 2)


    res = sum / (len(etalons_descriptors) - 1)
    return round(res, 2)


def get_list_of_dispersions(etalons_descriptors, degree: bool):
    list_of_dispersions = []
    # if I have 5 images -> I will have 2500 descriptors


    # etalon looks like if it has 2500 descriptors:
    # [0][0], [0][1], ..., [0][255]
    # [1][0],
    # ...
    # ...
    # [2499][0], ..., [2499][255]

    # need to get dispersion for each column of etalon.
    for i in range(len(etalons_descriptors[0])):
        list_of_dispersions.append(get_dispersion_for_etalon(etalons_descriptors, i, degree))

    # returns 256 dispersions
    return list_of_dispersions


# To divide each element of list
def divide_list_elements_by_number(list, number):
    divided_list = []
    for i in range(len(list)):
        divided_list.append(list[i] / number)
    return divided_list


def convert_list_to_dictionary(list):
    dictionary = {}
    for i in range(len(list)):
        dictionary[i] = list[i]
    return dictionary


# Dictionary has key and value.
# This method allows sort dictionary by value descending.
def sort_dictionary_by_value(dictionary):
    # return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
    return dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))


def write_dictionary_to_csv(dictionary, filename):
    with open(f'{filename}.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


# Look at the method description below.
# This method executes the lower method for all template descriptors
def cut_down_etalon_by_dispersions_indexes(descriptors, indexes):
    shortcut_etalon = []
    for i in range(len(descriptors)):
        shortcut_etalon.append(cut_down_descriptor_by_dispersions_indexes(descriptors[i], indexes))
    return shortcut_etalon


# The descriptor consists of an array of 256 elements.
# This method allows you to reduce the number of elements to 16
# due to the fact that only certain elements remain, whose indices
# were entered in the method parameters
def cut_down_descriptor_by_dispersions_indexes(descriptor, indexes):
    shortcut_descriptor = []  # len 16 will be
    for i in range(16):
        shortcut_descriptor.append(descriptor[indexes[i]])
    return shortcut_descriptor


# hausdorf

# method returns minimum distance for one descriptor. Planning to compare
# one with array of descriptors
def get_minimum_distance_for_descriptor_among_many(descriptor, many, distance_type):
    if distance_type == 'manhattan':
        min_distance = 100000
        for i in range(len(many)):
            distance_between_descriptors = manhattan_distance(descriptor, many[i])
            if distance_between_descriptors < min_distance:
                min_distance = distance_between_descriptors
        return min_distance

    if distance_type == 'hamming':
        min_distance = 256
        for i in range(len(many)):
            distance_between_descriptors = my_hamming_distance(descriptor, many[i])
            if distance_between_descriptors < min_distance:
                min_distance = distance_between_descriptors
        return min_distance


# method returns minimum distance for each descriptor from many_A.
# compares each descriptors from many_A with all descriptors from many_B
def get_minimum_distance_for_each_many_descriptors_among_many(many_A, many_B, distance_type):
    array_of_min_distances = []
    for i in range(len(many_A)):
        array_of_min_distances.append(get_minimum_distance_for_descriptor_among_many(many_A[i], many_B, distance_type))
    return array_of_min_distances


# returns minimum between two arrays.
def get_maximum_between_two_arrays_of_minimums(array_A, array_B, nameA, nameB):
    max_A = max(array_A)
    print(f'Max at {nameA}: {max_A}')
    max_B = max(array_B)
    print(f'Max at {nameB}: {max_B}')
    print()
    return max_A if max_A > max_B else max_B


# to convert hamming distance to the value in range [0;1]
# def convert_my_hamming_distance_to_standard(number_to_convert, max_distance):
#     return number_to_convert / max_distance


# if hamming distance less than 0.2 => return true
# def is_descriptors_equal(descriptor_A, descriptor_B):
#     dis = my_hamming_distance(descriptor_A, descriptor_B) / 256
#     # print(dis)
#     return dis < 0.2


# to find minimal hamming distance for descriptor when its comparing with collection of descriptors
# def findMinimalHammingDistanceForDescriptor(descriptor, etalons):
#     min_distance: float = 257
#     index = 1
#     index_of_min_distance = 0
#     while index < len(etalons):
#         current_distance = my_hamming_distance(descriptor, etalons[index])
#         # print(f'{index}) Current distance: {current_distance}')
#         if min_distance >= current_distance > 0.0:
#             index_of_min_distance = index
#             min_distance = current_distance
#         index += 1
#     return index_of_min_distance


# modified hamming distance function to return amount of different numbers at collection
# before this method returned value between 0 and 1 (1110 and 1011 returned 0.5)
# now my method returns amount of different values (1110 and 1011 return 2)
def my_hamming_distance(first_value, second_value):
    return float(len(first_value)) * distance.hamming(first_value, second_value)


def manhattan_distance(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))
    # arr1 = np.array(a)
    # arr2 = np.array(b)
    # return np.sum(np.abs(arr1 - arr2))


def get_distances_between_sets_by_hausdorf(sets, names, distance_type):
    dictionary = {} # A-B : 95
    i = 0
    while i < len(sets):
        j = i + 1
        while j < len(sets):
            tempA = get_minimum_distance_for_each_many_descriptors_among_many(sets[i], sets[j], distance_type)
            print(tempA)
            tempB = get_minimum_distance_for_each_many_descriptors_among_many(sets[j], sets[i], distance_type)
            print(tempB)
            max = get_maximum_between_two_arrays_of_minimums(tempA, tempB, names[i], names[j])
            dictionary[f'{names[i]} - {names[j]}'] = max
            j = j + 1
        i = i + 1
    return dictionary


def get_index_of_closest_etalon_for_etalon(etalon, etalons_set, etalon_name, names, distance_type):
    # if distance_type == 1:
    list = []
    for i in range(len(etalons_set)):
        tempA = get_minimum_distance_for_each_many_descriptors_among_many(etalon, etalons_set[i], distance_type)
        print(tempA)
        tempB = get_minimum_distance_for_each_many_descriptors_among_many(etalons_set[i], etalon, distance_type)
        print(tempB)
        maximum = get_maximum_between_two_arrays_of_minimums(tempA, tempB, etalon_name, names[i])
        print(f'{etalon_name} - {names[i]}: {maximum}')
        list.append(maximum)
    # test = min(dictionary, key=dictionary.get)
    # print(type(min))
    return list.index(min(list))


def is_method_compare_correctly(etalons, names, method):
    sum = 0
    for i in range(len(etalons)):
        index = get_index_of_closest_etalon_for_etalon(etalons[i], etalons, names[i], names, method)
        print(f'Index of closest: {index}')
        if index == i:
            sum = sum + 1

    if sum == len(etalons):
        print(f'Method works correctly; {sum} / {len(etalons)}')
        return True
    else:
        print(f'{sum} / {len(etalons)}')
        return False


def main():
    img_A = cv2.imread("Images/Leicter_more_white.jpg")
    img_B = cv2.imread("Images/Liverpool_more_white.jpg")
    img_C = cv2.imread("Images/ManUnited.jpg")
    img_D = cv2.imread("Images/Milan.jpg")
    img_E = cv2.imread("Images/Raingers.jpg")
    img_B_side = cv2.imread("Images/Liverpool_more_white_rotate_30.jpg")

    orb = cv2.ORB_create(nfeatures=500)

    keypoints_etalon_A, descriptors_etalon_A = orb.detectAndCompute(img_A, None)
    keypoints_etalon_B, descriptors_etalon_B = orb.detectAndCompute(img_B, None)
    keypoints_etalon_C, descriptors_etalon_C = orb.detectAndCompute(img_C, None)
    keypoints_etalon_D, descriptors_etalon_D = orb.detectAndCompute(img_D, None)
    keypoints_etalon_E, descriptors_etalon_E = orb.detectAndCompute(img_E, None)
    keypoints_etalon_B_side, descriptors_etalon_B_side = orb.detectAndCompute(img_B_side, None)

    # img = cv2.drawKeypoints(img_liverpool, keypoints_liverpool, None)
    # cv2.imshow("leicester", img)
    # cv2.waitKey(0)

    descriptors_etalon_B_side_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_B_side)
    descriptors_etalon_A_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_A)
    descriptors_etalon_B_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_B)
    descriptors_etalon_C_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_C)
    descriptors_etalon_D_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_D)
    descriptors_etalon_E_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_E)

    descriptors_etalon_A_integers = convert_bit_format_descriptors_to_integers_format(descriptors_etalon_A_bit_format)
    # descriptors_etalon_A_integers = convert_descriptors_to_whole_numbers(descriptors_etalon_B_side_bit_format)
    descriptors_etalon_B_integers = convert_bit_format_descriptors_to_integers_format(descriptors_etalon_B_bit_format)
    descriptors_etalon_C_integers = convert_bit_format_descriptors_to_integers_format(descriptors_etalon_C_bit_format)
    descriptors_etalon_D_integers = convert_bit_format_descriptors_to_integers_format(descriptors_etalon_D_bit_format)
    descriptors_etalon_E_integers = convert_bit_format_descriptors_to_integers_format(descriptors_etalon_E_bit_format)


    descriptors_combined_etalon_integers = \
        descriptors_etalon_A_integers + descriptors_etalon_B_integers +\
        descriptors_etalon_C_integers + descriptors_etalon_D_integers +\
        descriptors_etalon_E_integers

    print(f'Etalon len = {len(descriptors_combined_etalon_integers)}')
    print()

    # print(get_list_of_dispersions(descriptors_combined_etalon_integers))

    list_of_dispersions = get_list_of_dispersions(descriptors_combined_etalon_integers, True)
    print(f'List of dispersions len = {len(list_of_dispersions)}')
    print(list_of_dispersions)
    print()

    max_dispersion = max(list_of_dispersions)
    list_of_dispersions_divided_by_max = divide_list_elements_by_number(list_of_dispersions, max_dispersion)
    # print(list_of_dispersions)
    print(list_of_dispersions_divided_by_max)

    dictionary_of_dispersions_divided_by_max = convert_list_to_dictionary(list_of_dispersions_divided_by_max)
    print(dictionary_of_dispersions_divided_by_max)

    sorted_dictionary_of_dispersions_divided_by_max = sort_dictionary_by_value(dictionary_of_dispersions_divided_by_max)
    print(sorted_dictionary_of_dispersions_divided_by_max)

    # write_dictionary_to_csv(sorted_dictionary_of_dispersions_divided_by_max, 'без_квадратов')
    # write_dictionary_to_csv(sorted_dictionary_of_dispersions_divided_by_max, 'с_квадратами')

    print()
    top_16_sorted_dictionary_of_dispersions_divided_by_max = \
        dict(itertools.islice(sorted_dictionary_of_dispersions_divided_by_max.items(), 16))

    top_16_indexes = list(top_16_sorted_dictionary_of_dispersions_divided_by_max.keys())
    print(top_16_indexes)

    print('-' * 100)
    print()

    descriptors_etalon_A_integers_cut = \
        cut_down_etalon_by_dispersions_indexes(descriptors_etalon_A_integers, top_16_indexes)
    descriptors_etalon_B_integers_cut = \
        cut_down_etalon_by_dispersions_indexes(descriptors_etalon_B_integers, top_16_indexes)
    descriptors_etalon_C_integers_cut = \
        cut_down_etalon_by_dispersions_indexes(descriptors_etalon_C_integers, top_16_indexes)
    descriptors_etalon_D_integers_cut = \
        cut_down_etalon_by_dispersions_indexes(descriptors_etalon_D_integers, top_16_indexes)
    descriptors_etalon_E_integers_cut = \
        cut_down_etalon_by_dispersions_indexes(descriptors_etalon_E_integers, top_16_indexes)

    set_bits = [
        descriptors_etalon_A_bit_format,
        descriptors_etalon_B_bit_format,
        descriptors_etalon_C_bit_format,
        descriptors_etalon_D_bit_format,
        descriptors_etalon_E_bit_format
    ]

    set_integers = [
        descriptors_etalon_A_integers,
        descriptors_etalon_B_integers,
        descriptors_etalon_C_integers,
        descriptors_etalon_D_integers,
        descriptors_etalon_E_integers
    ]

    set_integers_cut = [
        descriptors_etalon_A_integers_cut,
        descriptors_etalon_B_integers_cut,
        descriptors_etalon_C_integers_cut,
        descriptors_etalon_D_integers_cut,
        descriptors_etalon_E_integers_cut
    ]

    names = ["A", "B", "C", "D", "E"]
    test_names = ["A", "B"]

# integers 265
    # manhattan   ,    hamming
    # start_time = time.time()
    # print('timer started...')
    # distances_integers_by_hausdorf = get_distances_between_sets_by_hausdorf(set_integers, names, 'manhattan')
    # print(f'manhattan:')
    # print(distances_integers_by_hausdorf)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #

# bits 256
    # print()
    # start_time = time.time()
    # print('timer started...')
    # distances_bits_by_hausdorf = get_distances_between_sets_by_hausdorf(set_bits, test_names, 'hamming')
    # print(f'hamming:')
    # print(distances_bits_by_hausdorf)
    # print("--- %s seconds ---" % (time.time() - start_time))

# integers 16
#     start_time = time.time()
#     print('timer started...')
#     distances_integers_cut_by_hausdorf = get_distances_between_sets_by_hausdorf(set_integers_cut, names, 'manhattan')
#
#     print(f'manhattan for cut:')
#     print(distances_integers_cut_by_hausdorf)
#     print("--- %s seconds ---" % (time.time() - start_time))

    test = is_method_compare_correctly(set_integers_cut, names, 'manhattan')
    print(test)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
