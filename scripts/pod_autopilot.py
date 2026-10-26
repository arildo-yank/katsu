"""Piloto automático de um run num pod da RunPod, rodando no Mac: espera o train_sft acabar, baixa os
arquivos com conferência de SHA-256 e desliga o pod (Stop) pela REST v2, confirmando o EXITED.

  python3 scripts/pod_autopilot.py --pod 9xmis0681djxov --ssh root@69.145.85.89:14165 \
      --remote /root/KATSU/runs/gpu-context-sft-v2 --local runs/gpu-context-sft-v2 \
      --files model.ckpt sft-report.json metrics.jsonl config.json tokenizer.bpe SHA256SUMS --max-usd 2

Travas de dinheiro:
  - desliga quando o custo do pod (cost/h × tempo desde startedAt) passar de --max-usd, mesmo no meio do treino;
  - desliga se o SSH ficar 30 min sem responder com o pod ainda ligado;
  - desliga mesmo que o download falhe por 2 h seguidas.
O processo do pod é procurado com `pgrep -x train_sft`: `pgrep -f <caminho>` encontra a própria linha do
SSH e diria "rodando" para sempre (o piloto de 2026-10-26 perdeu o fim do run por isso).
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / "ARANDU/.env"
SSH_KEY = str(Path.home() / ".ssh/arandu_gpu")


class Pilot:
    def __init__(self, a):
        self.a = a
        self.host, self.port = a.ssh.rsplit(":", 1)
        self.log_path = Path(a.local).with_suffix(".autopilot.log")

    def log(self, msg):
        line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def ssh(self, cmd, timeout=60):
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-i", SSH_KEY, "-p", self.port,
                            self.host, cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout

    def runpod(self, path, body=None):
        key = next(l.split("=", 1)[1].strip() for l in ENV.read_text().splitlines() if l.startswith("RUNPOD_API_KEY="))
        req = urllib.request.Request(f"https://api.runpod.io/v2{path}", data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")

    def pod(self):
        try:
            return self.runpod(f"/pods/{self.a.pod}")
        except Exception as error:
            self.log(f"RunPod indisponível: {type(error).__name__}")
            return {}

    def spent(self, pod):
        try:
            started = datetime.fromisoformat(pod["startedAt"].replace("Z", "+00:00"))
            return float(pod["cost"]) * (datetime.now(timezone.utc) - started).total_seconds() / 3600
        except (KeyError, TypeError, ValueError):
            return None

    def stop(self, reason):
        self.log(f"desligando: {reason}")
        for _ in range(20):
            status = self.pod().get("status")
            if status in ("EXITED", "TERMINATED"):
                self.log(f"pod {self.a.pod} confirmado parado ({status}): sem cobrança de GPU")
                return True
            if status in ("RUNNING", "STARTING", "PROVISIONING"):
                try:
                    self.runpod(f"/pods/{self.a.pod}/action", {"action": "stop"})
                    self.log("Stop enviado à RunPod")
                except Exception as error:
                    self.log(f"Stop falhou: {type(error).__name__} {error}")
            time.sleep(30)
        self.log("ALERTA: não consegui confirmar o Stop em 10 min; confira o painel da RunPod")
        return False

    def download(self):
        local = Path(self.a.local)
        local.mkdir(parents=True, exist_ok=True)
        _, out = self.ssh(f"cd {self.a.remote} && ls {' '.join(self.a.files)} 2>/dev/null; sha256sum *.ckpt 2>/dev/null")
        present = [f for f in self.a.files if f in out.split()]
        hashes = {l.split()[1]: l.split()[0] for l in out.splitlines() if len(l.split()) == 2 and l.split()[1].endswith(".ckpt")}
        for name in present:
            target = local / name
            r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", SSH_KEY, "-P", self.port,
                                f"{self.host}:{self.a.remote}/{name}", str(target)], timeout=3600, stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                raise RuntimeError(f"scp falhou em {name}")
            if name in hashes:
                got = subprocess.run(["shasum", "-a", "256", str(target)], capture_output=True, text=True).stdout.split()[0]
                if got != hashes[name]:
                    raise RuntimeError(f"hash diferente em {name}")
            self.log(f"baixado{' e conferido' if name in hashes else ''}: {name}")
        return present

    def run(self):
        self.log(f"piloto ligado: pod {self.a.pod}, teto US$ {self.a.max_usd}")
        down_since = None
        while True:
            pod = self.pod()
            cost = self.spent(pod)
            if cost is not None and cost > self.a.max_usd:
                self.stop(f"custo do pod chegou a US$ {cost:.2f} (teto {self.a.max_usd})")
                return 2
            try:
                code, out = self.ssh("pgrep -x train_sft >/dev/null && echo RUNNING || echo STOPPED")
            except subprocess.TimeoutExpired:
                code, out = 255, ""
            if code == 0:
                down_since = None
                if "STOPPED" in out:
                    self.log("train_sft terminou no pod")
                    break
            else:
                down_since = down_since or time.time()
                if time.time() - down_since > 1800:
                    self.stop("SSH sem resposta há 30 min")
                    return 1
            time.sleep(60)
        started = time.time()
        while True:
            try:
                self.log(f"download completo: {self.download()}")
                break
            except Exception as error:
                self.log(f"download falhou: {error}")
                if time.time() - started > 7200:
                    self.log("ALERTA: 2 h sem conseguir baixar; desligando assim mesmo")
                    break
                time.sleep(120)
        self.stop("run terminado e baixado")
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pod", required=True, help="id do pod na RunPod")
    p.add_argument("--ssh", required=True, help="user@host:porta (SSH over exposed TCP)")
    p.add_argument("--remote", required=True)
    p.add_argument("--local", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--max-usd", type=float, required=True)
    sys.exit(Pilot(p.parse_args(argv)).run())


if __name__ == "__main__":
    main()
