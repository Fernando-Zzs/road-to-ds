def print_hi(name):
    print(f'Hi, {name}')


def testInput():
    name = input("What's your name?")
    sum = 100 + 100
    print('hello,%s' % name)
    print('sum = %d' % sum)


def testIfElse(score):
    if score >= 90:
        print('Excellent')
    else:
        if score >= 60:
            print('Good')
        else:
            print('Fail')


def testFor():
    sum = 0
    for number in range(0, 11, 2):
        sum = sum + number
    print(sum)


def testWhile():
    sum = 0
    number = 0
    while number < 11:
        sum = sum + number
        number = number + 1
    print(sum)


def testList():
    lists = ['a', 'b', 'c']
    lists.append('d')
    print(lists)
    lists.insert(0, 'm')
    print(lists)
    lists.pop()
    print(lists)
    print(len(lists))


def testTuple():
    tuples = ('x', 'y', 'z')
    print(tuples)


def testDictionary():
    dict = {'zhangsan': 100, 'lisi': 99}
    dict['wangwu'] = 98
    print(dict)
    dict.pop('lisi')
    print('zhangsan' in dict)
    print(dict.get('wangwu'))


def testSet():
    s = set(['a', 'b', 'c'])
    s.add('d')
    s.remove('b')
    print(s)
    print('c' in s)


if __name__ == '__main__':
    # print_hi('PyCharm')
    # testInput()
    # testIfElse(100)
    # testFor()
    # testWhile()
    # testList()
    # testTuple()
    # testDictionary()
    testSet()
