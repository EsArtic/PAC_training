from ptbtokenizer import PTBTokenizer

def main():
    old_label = open('../label_simple.txt')
    new_label = open('../tokenized_label.txt', 'w')
    tool = PTBTokenizer()

    count = 0
    container = ''
    vids = []
    for line in old_label:
        items = line.strip().split('\t')
        vid = items[0]
        sentence = items[1]
        vids.append(vid)
        container = container + '\n' + sentence
        count += 1
        
    sentences = tool.tokenize(container)
    # print str
    for i in xrange(len(vids)):
        new_label.write(vids[i] + '\t' + sentences[i+1]+'\n')

if __name__ == '__main__':
    main()