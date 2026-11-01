"""Testes contra o servidor real em loopback; inicie-o antes de executar."""
import json, unittest, os
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError

class HttpTests(unittest.TestCase):
    def request(self, body=None, path='/generate'):
        req=Request(os.environ.get('KATU_URL','http://127.0.0.1:8787')+path, None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
        with build_opener(ProxyHandler({})).open(req,timeout=60) as r:return json.load(r)
    def test_health(self):
        r=self.request(path='/health');self.assertGreater(r['parameters'],2000000);self.assertEqual(r['backend'],'cpu')
    def test_generation_is_seeded_and_bounded(self):
        request={'prompt':'Olá, treinador!','max_new_tokens':4,'temperature':0.7,'seed':42}
        a=self.request(request);b=self.request(request)
        self.assertEqual(a['token_ids'],b['token_ids']);self.assertLessEqual(a['generated_tokens'],4)
        self.assertIsInstance(a['text'],str)
    def test_invalid_requests(self):
        for body in [{'prompt':''},{'prompt':'a','max_new_tokens':513},{'prompt':'a','temperature':-1},{'prompt':'a','repetition_penalty':0.5}]:
            with self.assertRaises(HTTPError) as c:self.request(body)
            self.assertEqual(c.exception.code,400)
    def test_sampling_defaults(self):
        r=self.request({'prompt':'Olá','max_new_tokens':1})
        self.assertAlmostEqual(r['temperature'],0.7)
        self.assertAlmostEqual(r['repetition_penalty'],1.2)
    def test_truncation_is_visible(self):
        r=self.request({'prompt':'jogador '*1000,'max_new_tokens':1})
        self.assertTrue(r['prompt_truncated']);self.assertGreater(r['prompt_tokens'],r['context_tokens'])

if __name__=='__main__':unittest.main()
