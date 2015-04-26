import sys
from collections import defaultdict

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    except Exception:
        print 'usage:: python %s <input_file> <output_file>' % sys.argv[0]
        sys.exit(1)

    author_names_to_author_ids = {}
    coauthorship_counts = defaultdict(int)

    author_id = 0
    for line in open(input_file):
        line = line.strip().split('\t')
        author_one = line[0]
        author_two = line[1]

        if author_one in author_names_to_author_ids:
            author_one_id = author_names_to_author_ids[author_one]
        else:
            author_one_id = author_id
            author_names_to_author_ids[author_one] = author_one_id
            author_id += 1

        if author_two in author_names_to_author_ids:
            author_two_id = author_names_to_author_ids[author_two]
        else:
            author_two_id = author_id
            author_names_to_author_ids[author_two] = author_two_id
            author_id += 1

        coauthorship_counts[(author_one_id, author_two_id)] += 1

    mapping_file = open(output_file + '_mapping.txt', 'w')
    graph_file = open(output_file + '.txt', 'w')

    for author_one_id, author_two_id in coauthorship_counts.keys():
        graph_file.write('%s,%s,%s\n' % (author_one_id, author_two_id,
                                         coauthorship_counts[(author_one_id, author_two_id)]))
        graph_file.write('%s,%s,%s\n' % (author_two_id, author_one_id,
                                         coauthorship_counts[(author_one_id, author_two_id)]))

    graph_file.close()

    for author_name in author_names_to_author_ids.keys():
        mapping_file.write('%s\t%s\n' % (author_name, author_names_to_author_ids[author_name]))

    mapping_file.close()