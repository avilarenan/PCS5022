"""Resume-safe immutable partitioned Parquet utility storage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utility_peft.types import UtilityRecord
from utility_peft.utils import canonical_json, stable_hash


class UtilityStore:
    """One immutable record per file, partitioned by dataset and horizon."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def append(self, record: UtilityRecord) -> bool:
        target = self.path_for(record)
        if target.exists():
            existing = self._read_file(target)
            if existing.key != record.key:
                raise RuntimeError(f"Hash collision at {target}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        row = record.to_flat_dict()
        row["evidence_json"] = canonical_json(row.pop("evidence"))
        table = pa.Table.from_pylist([row])
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(table, temporary, compression="zstd")
        if target.exists():
            temporary.unlink()
            return False
        temporary.replace(target)
        return True

    def path_for(self, record: UtilityRecord) -> Path:
        dataset = record.dataset.replace("/", "_")
        filename = stable_hash(record.key, length=32) + ".parquet"
        return self.root / f"dataset={dataset}" / f"horizon={record.horizon}" / filename

    def records(
        self,
        *,
        datasets: set[str] | None = None,
        episode_ids: set[str] | None = None,
        action_ids: set[str] | None = None,
        statuses: set[str] | None = None,
        config_hash: str | None = None,
        model_revision: str | None = None,
    ) -> list[UtilityRecord]:
        records: list[UtilityRecord] = []
        for path in sorted(self.root.glob("dataset=*/horizon=*/*.parquet")):
            record = self._read_file(path)
            if datasets is not None and record.dataset not in datasets:
                continue
            if episode_ids is not None and record.episode_id not in episode_ids:
                continue
            if action_ids is not None and record.action_id not in action_ids:
                continue
            if statuses is not None and record.status not in statuses:
                continue
            if config_hash is not None and record.config_hash != config_hash:
                continue
            if model_revision is not None and record.model_revision != model_revision:
                continue
            records.append(record)
        return records

    def _read_file(self, path: Path) -> UtilityRecord:
        table = pq.ParquetFile(path).read()
        if table.num_rows != 1:
            raise RuntimeError(f"Immutable utility file must contain one row: {path}")
        row = table.to_pylist()[0]
        row["evidence"] = json.loads(row.pop("evidence_json"))
        return UtilityRecord(**row)
