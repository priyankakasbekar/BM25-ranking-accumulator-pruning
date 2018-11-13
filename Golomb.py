import math
import random

def LengthRice(m,k):

    quotient = math.floor((k-1)/m)

    num_of_bits = 0

    num_of_bits += quotient+1

    remainder = (k-1) % m

    if len(bin(remainder)[2:]) < math.floor(math.log2(m)):
        num_of_bits += math.floor(math.log2(m))
    elif len(bin(remainder)[2:]) == math.floor(math.log2(m)):
        num_of_bits += math.floor(math.log2(m))
    elif len(bin(remainder)[2:]) == math.ceil(math.log2(m)):
        num_of_bits += math.ceil(math.log2(m))

    return num_of_bits

def LengthGolomb(m,k):
    quotient = math.floor((k - 1) / m)
    num_of_bits = 0
    num_of_bits += quotient + 1
    remainder = (k - 1) % m

    range = (2 ** math.ceil(math.log2(m))) - m - 1


    if remainder <= range:
        num_of_bits += math.floor(math.log2(m))
    else:
        num_of_bits += math.ceil(math.log2(m))

    return num_of_bits

def CalcProbabilityGap(gap):
    return ((1-(1/163))**(gap-1)) * (1/163)

def GenerateGap():

    t = 1

    while random.uniform(0.0,0.9) <= (1/163):
        t += 1

    return t

def GenerateGapLst(size):

    gap_lst = []
    count = size

    while count > 0:
        gap_lst.append(GenerateGap())
        count -= 1

    return gap_lst

def FindAverageLength():

    size = 145
    gap_lst = GenerateGapLst(size)
    m_golomb = 114
    m_rice = 128

    total_len_golomb = 0
    total_len_rice = 0
    total_gap_probability = 1

    for gap in gap_lst:
        total_len_golomb += LengthGolomb(m_golomb,gap)
        total_len_rice += LengthRice(m_rice,gap)
        total_gap_probability *= float(CalcProbabilityGap(gap))

    print("Golomb Avg Length: ", (total_len_golomb/size))
    print("Rice Avg Length: ", (total_len_rice / size))
    print("Arithmetic Code Avg Length: ",float(math.ceil(-(math.log2(total_gap_probability))) + 1)/size)


FindAverageLength()