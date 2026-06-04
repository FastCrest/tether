from tether.verification_report import _HASH_CHUNK, _sha256


def test_sha256_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert _sha256(path) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_small_file(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"abc")

    assert _sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_large_file(tmp_path):
    path = tmp_path / "large.bin"
    payload = b"tether-vla-sha256-test\n" * (
        (_HASH_CHUNK // len(b"tether-vla-sha256-test\n")) + 3
    )
    path.write_bytes(payload)

    assert len(payload) > _HASH_CHUNK
    assert _sha256(path) == (
        "be4d788a195785d6e979d2958e8095ec19af664d4413416b474dc0e4f7a15c24"
    )
