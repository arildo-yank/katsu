import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import export_to_gguf as exporter

# Leitor independente e mínimo dos tipos emitidos; não usa o writer para validar offsets.
def read_gguf(path):
    with Path(path).open('rb') as file:
        def num(fmt):
            return struct.unpack('<'+fmt,file.read(struct.calcsize('<'+fmt)))[0]
        def string():
            return file.read(num('Q')).decode('utf-8')
        def value(kind):
            if kind==8:return string()
            if kind==9:
                subtype=num('I');count=num('Q');return [value(subtype) for _ in range(count)]
            return num({4:'I',10:'Q',6:'f',7:'?'}[kind])
        assert file.read(4)==b'GGUF'
        assert num('I')==3
        count=num('Q');kv=num('Q');metadata={}
        for _ in range(kv):
            key=string();metadata[key]=value(num('I'))
        tensors=[]
        for _ in range(count):
            name=string();rank=num('I');dims=[num('Q') for _ in range(rank)]
            dtype=num('I');offset=num('Q');tensors.append((name,dims,dtype,offset))
        start=(file.tell()+31)//32*32
        values=[]
        for name,dims,dtype,offset in tensors:
            assert offset%32==0 and dtype==0
            length=1
            for d in dims:length*=d
            file.seek(start+offset);values.append((name,dims,file.read(length*4)))
        return metadata,values

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        # 256 bytes + merge aa, no artificial vocabulary/token normalization.
        (self.root/'tokenizer.bpe').write_bytes(b'KATSUBP1'+struct.pack('<III',1,97,97))
        config=dict(vocab_size=257,block_size=2,embed_dim=2,layers=0,heads=1)
        (self.root/'config.json').write_text(json.dumps(config))
        shapes=[[257,2],[2,2],[2],[2],[2,257]]
        self.payloads=[]
        with (self.root/'model.ckpt').open('wb') as f:
            f.write(b'KATSUCK1'+struct.pack('<Q',5))
            for i,shape in enumerate(shapes):
                f.write(struct.pack('<Q',len(shape))+struct.pack('<'+'Q'*len(shape),*shape))
                n=1
                for d in shape:n*=d
                data=struct.pack('<'+'f'*n,*[float(i+j/1024) for j in range(n)])
                f.write(data);self.payloads.append(data)
    def tearDown(self):self.tmp.cleanup()
    def test_export_preserves_every_byte_and_bpe(self):
        output=self.root/'model.gguf';exporter.export(self.root,output)
        metadata,tensors=read_gguf(output)
        self.assertEqual(metadata['general.architecture'],'katu')
        self.assertEqual(metadata['tokenizer.katu.merge_ids'],[97,97])
        alphabet=exporter.byte_alphabet();inverse={c:b for b,c in alphabet.items()}
        decoded=[bytes(inverse[c] for c in text) for text in metadata['tokenizer.ggml.tokens']]
        self.assertEqual(decoded,[bytes([i]) for i in range(256)]+[b'aa'])
        self.assertEqual([t[2] for t in tensors],self.payloads)
        self.assertEqual(tensors[0][1],[2,257])
        self.assertEqual(tensors[-1][1],[257,2])
        with self.assertRaises(ValueError):exporter.export(self.root,output)
    def test_mismatched_vocab_rejected(self):
        config=json.loads((self.root/'config.json').read_text());config['vocab_size']=256
        (self.root/'config.json').write_text(json.dumps(config))
        with self.assertRaises(ValueError):exporter.export(self.root,self.root/'bad.gguf')
        self.assertFalse((self.root/'bad.gguf').exists())
    def test_truncated_checkpoint_rejected(self):
        p=self.root/'model.ckpt';p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):exporter.export(self.root,self.root/'bad.gguf')
    def test_reserved_tokens_roundtrip_in_gguf(self):
        p=self.root/'tokenizer.bpe';p.write_bytes(b'KATSUBP2'+p.read_bytes()[8:])
        c=json.loads((self.root/'config.json').read_text());c['vocab_size']=261
        (self.root/'config.json').write_text(json.dumps(c))
        layout=list(exporter.layout(c))
        with (self.root/'model.ckpt').open('wb') as f:
            f.write(b'KATSUCK1'+struct.pack('<Q',len(layout)))
            for _,dims in layout:
                f.write(struct.pack('<Q',len(dims))+struct.pack('<'+'Q'*len(dims),*dims))
                n=1
                for d in dims:n*=d
                f.write(struct.pack('<'+'f'*n,*([0.0]*n)))
        out=self.root/'special.gguf';exporter.export(self.root,out)
        m,_=read_gguf(out)
        self.assertEqual(m['tokenizer.katu.special_ids'],[257,258,259,260])
        self.assertEqual(m['tokenizer.ggml.eos_token_id'],260)
        self.assertEqual(m['tokenizer.ggml.tokens'][-1],'<|end|>')

    def test_bad_merge_rejected(self):
        (self.root/'tokenizer.bpe').write_bytes(b'KATSUBP1'+struct.pack('<III',1,999,1))
        with self.assertRaises(ValueError):exporter.load_bpe(self.root/'tokenizer.bpe')

if __name__=='__main__':unittest.main(verbosity=2)
