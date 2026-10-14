# Legacy backup verification — 2026-10-12

2026-10-14 update: the user's newer explicit instruction authorized creating a
new home first and deleting the legacy GitHub repository immediately. After a
fresh33-commit bundle restore/fsck, remote-ref match and checkpoint SHA/CRC
verification, the old repository was deleted and authenticated API404 confirmed.
The new public home is MaYangle/Multimodal-Sentiment-Fusion. Historical gates
below are superseded for this repository operation; research completion remains
pending. [Exact migration and backup scope](GITHUB_MIGRATION_20261014.md).

The earlier audit referenced `D:/MYL/JevAffect-backup/legacy-source`; that
directory is absent on this machine now. Its existence cannot support current
cleanup authorization. A new backup was obtained from the actual remote
`https://github.com/MaYangle/Multimodal-Sentiment-Analysis.git`.

## Verified Git/source artifacts

The remote advertises exactly two branches and no tags. Fresh `clone --mirror`
is non-shallow, has33 commits reachable across all refs, and passes
`git fsck --full`. Independently checked `git ls-remote` matches:

| Ref | SHA |
|---|---|
| HEAD / main | f6f9aaf6d80f596e3600fca0aa0d452cfcca0414 |
| Memor | 22a53e35864b281eb65360bc86a27f7c5e0d4a80 |

`legacy-all.bundle` records complete history with no prerequisites. It was
actually cloned into a separate restore-check mirror; restored refs match and
all restored Git objects pass fsck. The main source tar is separately exported.
Durable copies under ignored `backups/legacy_20261012` are re-hashed/verified.

| Artifact | SHA256 |
|---|---|
| legacy-all.bundle | bc3ce92644c91c1a940504de3dd26e81bda590cddb85d9bc28eca53ea95e8c8a |
| legacy-main-source.tar | 06b4bdf4028d24179811756c42cbd9ac5aaf34db5cdee3428ae42305eca4bf2b |

Git includes no reachable `.pt`, `.pth`, `.pkl`, `.bin` or `.safetensors` payload
files. Source, history, LICENSE and README data/model references are retained
by the bundle and tar. The current main still calls training DataLoader with
`force_test_mode=True` and subsequently expects `dataloaders['train']`; its
external `best_models/zh.pt` and `en.pt` paths remain. This revalidates key
findings of the historical [static code audit](legacy_audit.md), while its old
filesystem path is not represented as a current asset.

## Model-source backup is still incomplete

The archived main README explicitly publishes Drive folder
`1deCsD3TXacpuov78v7PldhXL5zFSjNjL`. A fresh listing records `best_models.zip`,
`pretrained_models.rar`, `MSA.zip`, `Experimental report.pdf` and README.
`best_models.zip` ID `1tNrwyv5pCelgEN8r1nt2pz_BtLDiOYZm` has126,090,787 bytes;
`pretrained_models.rar` ID `1K82B5rsga_JnKmnwvg3p92u8JoeAbYGh` has4,535,034,039.
Finite responses have exact total lengths and Last-Modified; transfer caches
and source receipts are in ignored `work/legacy_backup_20261012`. After a network
timeout, best_models.zip resumed and completed; SHA256
`4faf129c1cc15f80c2faa0a32647d2f8bea4f97242e038143ce86fa5f551a5c2`.
All ZIP entries pass CRC and safe-path inspection: en.pt and zh.pt, each
72,998,565 uncompressed bytes. The [weight archive audit](legacy_best_models_backup.json)
records sizes/CRCs; archive and receipt also have durable local backup copies.
No old model code or weights were executed. The additional pretrained archive
remains incomplete; its initial handle has terminated with an unsuccessful
child transfer. Revalidate availability/cache before any resume.
Published links are preserved without granting redistribution rights.

## Cleanup gate

The verified Git backup alone does not complete all requested backup/release
conditions. Obtain available checkpoint archives, verify their integrity and
retain source/usage records; complete the new project's research, fresh-environment
reproduction and GitHub publication; then revalidate remote refs/backup state
before deletion. The old repository remains intact and has not been deleted.
