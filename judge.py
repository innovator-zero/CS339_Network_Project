def judge(data_send, data_receive):
    f1 = open(data_send, 'r')
    s1 = f1.read()
    s1_valid = s1[0:2440]

    f2 = open(data_receive, 'r')
    s2 = f2.read()

    print(len(s1_valid))
    print(len(s2))

    wrong = False
    num = 0
    for i in range(len(s1_valid)):
        if s1_valid[i] != s2[i]:
            num += 1
            wrong = True


    if not wrong:
        print('Right!')
    else:
        print('Wrong:' + str(num))

judge('data_input.txt', 'output.txt')
