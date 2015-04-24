from xml.etree.ElementTree import iterparse, XMLParser
import htmlentitydefs


class CustomEntity:
    def __getitem__(self, key):
        if key == 'umml':
            key = 'uuml'  # Fix invalid entity
        return unichr(htmlentitydefs.name2codepoint[key])


if __name__ == '__main__':
    confs = ['KDD', 'SDM', 'CIKM', 'SIGMOD Conference', 'ICDM', 'VLDB', 'PVLDB']
    out_dir = '/Users/pratik/Research/datasets/DBLP/coauthorship/'
    parser = XMLParser()
    parser.parser.UseForeignDTD(True)
    parser.entity = CustomEntity()

    fv = open(out_dir + 'VLDB_05_13.txt', 'w')
    fk = open(out_dir + 'KDD_05_13.txt', 'w')
    fs = open(out_dir + 'SDM_05_13.txt', 'w')
    fc = open(out_dir + 'CIKM_05_13.txt', 'w')
    fm = open(out_dir + 'SIGMOD_05_13.txt', 'w')
    fi = open(out_dir + 'ICDM_05_13.txt', 'w')

    count = 0
    for (event, node) in iterparse('/Users/pratik/Research/datasets/DBLP/dblp.xml', events=['start'], parser=parser):
        if node.tag == 'inproceedings':
            book_title = node.find('booktitle')

            if book_title is not None and book_title.text in confs:
                conf = book_title.text
                year = node.find('year')

                if year is not None:
                    year = int(year.text)

                    if 2005 <= year <= 2013:
                        authors = node.findall('author')
                        author_names = []

                        for author in authors:
                            author_names.append(author.text)

                        author_names = sorted(author_names)
                        no_authors = len(author_names)

                        if no_authors > 1:
                            for i in xrange(no_authors - 1):
                                for j in xrange(i + 1, no_authors):
                                    if conf == 'SDM':
                                        fs.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))
                                    if conf == 'VLDB' or conf == 'PVLDB':
                                        fv.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))
                                    if conf == 'CIKM':
                                        fc.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))
                                    if conf == 'ICDM':
                                        fi.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))
                                    if conf == 'KDD':
                                        fk.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))
                                    if conf == 'SIGMOD Conference':
                                        fm.write('%s\t%s\n' % (author_names[i].encode('utf-8'),
                                                               author_names[j].encode('utf-8')))

        node.clear()

        count += 1
        if count % 100000 == 0:
            print 'Processed Record: ', count

    fs.close()
    fc.close()
    fi.close()
    fk.close()
    fm.close()
    fv.close()
