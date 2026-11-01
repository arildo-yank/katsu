"""Gera a lista de componentes do Katu (SBOM, CycloneDX 1.5) a partir do Cargo.lock.

Cada crate de terceiros entra com versão, licença, o SHA-256 do Cargo.lock e quais crates do
workspace a usam (só dependências normais, com todas as features: inclui o CUDA). Sem data nem
caminho da máquina: o mesmo Cargo.lock gera o mesmo arquivo.

    python3 scripts/soberania/sbom.py           # grava docs/soberania/sbom.json
    python3 scripts/soberania/sbom.py --check   # falha se o arquivo estiver desatualizado
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "soberania" / "sbom.json"


def metadata():
    out = subprocess.run(
        ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def build():
    meta = metadata()
    packages = {p["id"]: p for p in meta["packages"]}
    members = set(meta["workspace_members"])
    # Grafo resolvido, só arestas de dependência normal (nada de build/dev).
    edges = {
        n["id"]: [d["pkg"] for d in n["deps"] if any(k["kind"] is None for k in d["dep_kinds"])]
        for n in meta["resolve"]["nodes"]
    }
    used_by = {}
    for member in sorted(members, key=lambda i: packages[i]["name"]):
        seen, stack = set(), [member]
        while stack:
            for dep in edges[stack.pop()]:
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        for dep in seen - members:
            used_by.setdefault(dep, []).append(packages[member]["name"])

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in lock["package"]}

    components = []
    for pid in sorted(used_by, key=lambda i: (packages[i]["name"], packages[i]["version"])):
        p = packages[pid]
        checksum = checksums.get((p["name"], p["version"]))
        if not checksum:
            raise SystemExit(f"{p['name']} {p['version']} não tem checksum no Cargo.lock")
        components.append({
            "type": "library",
            "name": p["name"],
            "version": p["version"],
            "purl": f"pkg:cargo/{p['name']}@{p['version']}",
            "licenses": [{"expression": p.get("license") or "NOASSERTION"}],
            "hashes": [{"alg": "SHA-256", "content": checksum}],
            "properties": [{"name": "katu:used-by", "value": ",".join(used_by[pid])}],
        })
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": "katu", "licenses": [{"license": {"id": "MIT"}}]}},
        "components": components,
    }


def main():
    bom = build()
    count = len(bom["components"])
    text = json.dumps(bom, indent=2, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            raise SystemExit(f"{OUT.relative_to(ROOT)} desatualizado: rode python3 scripts/soberania/sbom.py")
        print(f"SBOM em dia: {count} componentes")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"{OUT.relative_to(ROOT)}: {count} componentes")


if __name__ == "__main__":
    main()
