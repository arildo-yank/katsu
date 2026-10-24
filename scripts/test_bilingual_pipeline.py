import unittest
from build_bilingual_corpus import parse_article, trim

class CorpusTests(unittest.TestCase):
    def xml(self,license='https://creativecommons.org/licenses/by/4.0/',lang='en'):
        return f'''<article xml:lang="{lang}"><front><article-meta><title-group><article-title>Sports research</article-title></title-group><permissions><license href="{license}">License</license></permissions></article-meta></front><body><p>{'Movement and recovery in soccer. '*100}</p><fig><p>FIGURE MUST NOT ENTER TEXT</p></fig></body></article>'''.encode()
    def test_license_and_body(self):
        s,m=parse_article(self.xml());self.assertNotIn('FIGURE',s);self.assertEqual(m['license'],'https://creativecommons.org/licenses/by/4.0/')
    def test_rejects_restricted_and_missing_license(self):
        for lic in ['https://creativecommons.org/licenses/by-nc/4.0/','https://creativecommons.org/licenses/by-nd/4.0/','']:
            with self.assertRaises(ValueError):parse_article(self.xml(lic))
    def test_rejects_other_language(self):
        with self.assertRaises(ValueError):parse_article(self.xml(lang='pt'))
    def test_utf8_cut_never_emits_broken_bytes(self):
        for n in range(10):trim('prontidão'.encode(),n).decode('utf-8')
if __name__=='__main__':unittest.main()
