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


def converted_descriptor_by_adamar_matrix(descriptor, adamar_matrix):
    result_array = []
    for i in range(len(descriptor)):
        sum = 0
        for j in range(len(descriptor)):
            sum = sum + (descriptor[j] * adamar_matrix[i][j])
        result_array.append(sum)
    return result_array


def convert_descriptors_to_whole_numbers(descriptors_bit_format):
    adamar_matrix = hadamard(256)
    converted_descriptors = []
    for i in range(len(descriptors_bit_format)):
        # print(i)
        converted_descriptors.append(converted_descriptor_by_adamar_matrix(descriptors_bit_format[i], adamar_matrix))
    return converted_descriptors


def get_dispersion_for_etalon(etalon, index):
    sum_for_average = 0
    average = 0

    for j in range(len(etalon)): # range 1000
        sum_for_average = sum_for_average + etalon[j][index]
    average = sum_for_average / len(etalon)

    sum = 0
    for j in range(len(etalon)):
         sum = sum + ((etalon[j][index] - average) ** 2)

    res = sum / (len(etalon) - 1)
    return round(res, 2)


def get_dispersion_for_etalon(etalon, index, degree):
    sum_for_average = 0
    average = 0

    for j in range(len(etalon)): # range 1000
        sum_for_average = sum_for_average + etalon[j][index]
    average = sum_for_average / len(etalon)

    sum = 0
    if degree:
        for j in range(len(etalon)):
             sum = sum + (((etalon[j][index] ** 2) - (average ** 2)) ** 2)
    else:
        for j in range(len(etalon)):
             sum = sum + ((etalon[j][index] - average) ** 2)

    res = sum / (len(etalon) - 1)
    return round(res, 2)


def get_list_of_dispersions(etalon, degree: bool):
    list_of_dispersions = []
    for i in range(len(etalon[0])):
        list_of_dispersions.append(get_dispersion_for_etalon(etalon, i, degree))

    return list_of_dispersions


def divide_list_elements_by_number(list, number):
    divided_list = []
    for i in range(len(list)):
        divided_list.append(list[i] / number)
    return divided_list


def convert_list_to_dictionary(list):
    dictionay = {}
    for i in range(len(list)):
        dictionay[i] = list[i]
    return dictionay

def sort_dictionary_by_value(dictionary):
    # return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
    return dict( sorted(dictionary.items(), key=operator.itemgetter(1),reverse=True))


def write_dictionary_to_csv(dictionary, filename):
    with open(f'{filename}.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def main():
    img_A = cv2.imread("Images/Leicter_more_white.jpg")
    img_B = cv2.imread("Images/Liverpool_more_white.jpg")
    img_B_side = cv2.imread("Images/Liverpool_more_white_rotate_30.jpg")

    orb = cv2.ORB_create(nfeatures=500)

    keypoints_etalon_A, descriptors_etalon_A = orb.detectAndCompute(img_A, None)
    keypoints_etalon_B, descriptors_etalon_B = orb.detectAndCompute(img_B, None)
    keypoints_etalon_B_side, descriptors_etalon_B_side = orb.detectAndCompute(img_B_side, None)

    # img = cv2.drawKeypoints(img_liverpool, keypoints_liverpool, None)
    # cv2.imshow("leicester", img)
    # cv2.waitKey(0)

    descriptors_etalon_B_side_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_B_side)
    descriptors_etalon_B_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_B)
    descriptors_etalon_A_bit_format = convert_32_descriptors_to_256_bit(descriptors_etalon_A)


    descriptors_etalon_A_whole_numbers = convert_descriptors_to_whole_numbers(descriptors_etalon_A_bit_format)
    descriptors_etalon_B_whole_numbers = convert_descriptors_to_whole_numbers(descriptors_etalon_B_bit_format)


    descriptors_combined_etalon_whole_numbers = descriptors_etalon_A_whole_numbers + descriptors_etalon_B_whole_numbers

    # print(get_list_of_dispersions(descriptors_combined_etalon_whole_numbers))

    list_of_dispersions = get_list_of_dispersions(descriptors_combined_etalon_whole_numbers, False)
    max_dispersion = max(list_of_dispersions)
    list_of_dispersions_divided_by_max = divide_list_elements_by_number(list_of_dispersions, max_dispersion)
    # print(list_of_dispersions)
    print(list_of_dispersions_divided_by_max)
    dictionary_of_dispersions_divided_by_max = convert_list_to_dictionary(list_of_dispersions_divided_by_max)
    print(dictionary_of_dispersions_divided_by_max)

    sorted_dictionary_of_dispersions_divided_by_max = sort_dictionary_by_value(dictionary_of_dispersions_divided_by_max)
    print(sorted_dictionary_of_dispersions_divided_by_max)

    write_dictionary_to_csv(sorted_dictionary_of_dispersions_divided_by_max, 'без_квадратов')

    # print(len(descriptors_etalon_A_whole_numbers))
    # print(descriptors_etalon_A_whole_numbers[0])

    # etalon_liverpool_plus_lester = []
    #
    # etalon_liverpool_plus_lester_bit_format = descriptors_liverpool_bit_format + descriptors_lester_bit_format
    #
    # for liverpool_value in range(len(descriptors_liverpool)):
    #     etalon_liverpool_plus_lester.append(descriptors_liverpool[liverpool_value])
    #
    # for lester_value in range(len(descriptors_lester)):
    #     etalon_liverpool_plus_lester.append(descriptors_lester[lester_value])


    # print(descriptors_liverpool_bit_format[0])


    # converted_descriptor =

    # print(len(converted_descriptor))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
