from pyspark import SparkContext, SparkConf
import sys
import os

def main():
    assert len(sys.argv) == 5, "Usage: python G020HW1.py <K> <H> <S> <file_name>"

    conf = SparkConf().setAppName('G030HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions K
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read H
    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    # 3. Read string S
    S = sys.argv[3]
    assert isinstance(S, str), "S must be a string"
    S = str(S)

    # 4. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions=K).cache()
    rawData.repartition(numPartitions=K)

    # TASK 1.
    nRows = rawData.count()
    print("\n- Number of rows = ", nRows)
    #print(rawData.collect())


    # TASK 2.
    rawData = rawData.map(lambda line: line.split(','))
    #print(rawData.collect())

    if S == 'all':
        rawDataFiltered = rawData.filter(lambda x: int(x[3]) > 0)
    else:
        rawDataFiltered = rawData.filter(lambda x: int(x[3]) > 0)
        rawDataFiltered = rawDataFiltered.filter(lambda s: str(s[7]) == S)

    productCustomer = (rawDataFiltered.map(lambda x: (x[1], x[6]))
                       .groupByKey()
                       .map(lambda x: (x[0], list(x[1])))
                       .mapValues(lambda x: set(x))
                       .flatMap(lambda l: [(l[0], v) for v in l[1]]))

    nRowsPC = productCustomer.count()
    print(productCustomer.collect())
    print("\n- Product-Customer Pairs = ", nRowsPC)  # <-- Print the n. of lines of productCustomer

    # TASK 3.
    def f(x):
        for i in x:
            yield (i)

    productPopularity1 = (productCustomer
                          .mapPartitions(f)
                          .groupByKey()
                          .mapValues(lambda x: len(x)))
    # TASK 4.
    productPopularity2 = (productCustomer
                          .map(lambda x: (x[0], 1))
                          .reduceByKey(lambda x, y: x + y))

    # TASK 5.
    if H > 0:
        productHighestPopularity = (productPopularity2.sortBy(lambda x: x[1], ascending=False)
                                    .take(H))

        print('\n- Top', H, 'Products and their Popularity:')
        print(productHighestPopularity, '\n')

    # TASK 6.
    if H == 0:
        productPopularity1List = productPopularity1.sortByKey().collect()
        productPopularity2List = productPopularity2.sortByKey().collect()

        print('\n- productPopularity1 in increasing lexicographic order:')
        print(productPopularity1List)
        print('\n- productPopularity2 in increasing lexicographic order:')
        print(productPopularity2List, '\n')


if __name__ == "__main__":
    main()
