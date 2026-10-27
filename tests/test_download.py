import hashlib

import pytest

from boxsmith.fetch_checkpoint import download


def test_verified_download_and_refuse_overwrite(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"example checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "download" / "best.pt"
    assert download(output, source.as_uri(), digest).read_bytes() == source.read_bytes()
    assert download(output, source.as_uri(), digest) == output
    with pytest.raises(FileExistsError):
        download(output, source.as_uri(), "0" * 64)
    with pytest.raises(ValueError):
        download(tmp_path / "corrupted.pt", source.as_uri(), "0" * 64)
    assert not (tmp_path / "corrupted.pt").exists()
