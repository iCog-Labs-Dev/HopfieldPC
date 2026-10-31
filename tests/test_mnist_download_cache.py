import gzip
import struct
from pathlib import Path

import pytest

from fabricpc.hopfield.benchmarks import mnist_baseline as mb


def _tiny_mnist_images_gzip() -> bytes:
    idx_data = struct.pack(">IIII", 2051, 1, 28, 28) + bytes(28 * 28)
    return gzip.compress(idx_data)


def test_corrupt_mnist_cache_is_replaced_with_valid_download(tmp_path, monkeypatch):
    cache_path = tmp_path / "train-images-idx3-ubyte.gz"
    cache_path.write_bytes(b"truncated gzip")
    expected_download = _tiny_mnist_images_gzip()
    downloaded_urls = []

    def fake_urlretrieve(url, destination):
        downloaded_urls.append(url)
        Path(destination).write_bytes(expected_download)
        return str(destination), None

    monkeypatch.setattr(mb.urllib.request, "urlretrieve", fake_urlretrieve)

    mb._ensure_mnist_file(
        cache_path,
        "https://example.test/mnist-images.gz",
        image_file=True,
        expected_count=1,
    )

    assert downloaded_urls == ["https://example.test/mnist-images.gz"]
    assert cache_path.read_bytes() == expected_download
    assert list(tmp_path.iterdir()) == [cache_path]


def test_invalid_mnist_download_is_not_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "train-images-idx3-ubyte.gz"

    def fake_urlretrieve(url, destination):
        Path(destination).write_bytes(gzip.compress(b"not an IDX file"))
        return str(destination), None

    monkeypatch.setattr(mb.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(
        RuntimeError, match="Downloaded MNIST file is incomplete or invalid"
    ):
        mb._ensure_mnist_file(
            cache_path,
            "https://example.test/mnist-images.gz",
            image_file=True,
            expected_count=1,
        )

    assert not cache_path.exists()
    assert list(tmp_path.iterdir()) == []
