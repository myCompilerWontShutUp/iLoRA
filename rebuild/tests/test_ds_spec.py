"""데이터셋 규격 (DESIGN.md §23.2, §23.4, §23.7 (a)): 값 표, 별칭, 입력 해시 표, MovieLens 프로파일 golden."""
from __future__ import annotations

import dataclasses
import hashlib
import types
from pathlib import Path

import numpy as np
import pytest

from ilora import batching, cluster, data, evaluate, prompts, run, sections
from ilora.config import config_hash, dump_config, load_config
from ilora.datasets import LASTFM, MOVIELENS, SPECS, STEAM, DatasetError, DatasetSpec, for_template, spec_of

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
CONFIG_DIR = REBUILD_DIR / "configs"
ARTIFACT_SET = "cluster.artifact_dir=runs/rebuild/cluster_run_for_tests/cluster"
NEEDS_ARTIFACT = ("cluster_hard", "variant_user_level_cluster")

# D39 (§23.2 table of bracketed titles): slot and special token strings, single brackets are allowed.
ID2NAME_FORBIDDEN = ("[HistoryHere]", "[CansHere]", "[HistoryEmb]", "[CansEmb]", "[PH]", "[ItemEmb]", "[PAD]",
                     "<s>", "</s>", "<unk>")
# §23.2 value table (the rows that DatasetSpec holds); MovieLens fingerprints are §5.6.9.
TABLE = {
    "movielens": {
        "data_dir": "data/ref/movielens", "item_file": "u.item", "item_format": "uitem", "num_items": 1682,
        "has_ratings": True, "first_window_empty": True, "extra_separator_ids": (),
        "expected_rows": {"train": 68388, "val": 94, "test": 95},
        "expected_samples": {"train": 66126, "val": 94, "test": 95},
        "train_blocks": 754, "train_users": 754,
        "fingerprints": {"train": "85bafab832f1f78e5095b1fd880a418097054d3ea7dbb7d4fe9e1528f019aa99",
                         "val": "de7ac66b6703dc6935e1339183c00e08af2ad40f6d0877e7556b483a9aa65523",
                         "test": "01636285c16faf95cc7136bc937b9aa315c66de5e23fed5efec0e2b2e84dff09"},
        "template_file": "prompt/movie.txt", "sasrec_source": "rec_model/movielens.pt",
        "title_modes": ("code", "fixed"), "title_forbidden": ("[", "]", "<s>", "</s>", "<unk>"),
    },
    "steam": {
        "data_dir": "data/ref/steam", "item_file": "id2name.txt", "item_format": "id2name", "num_items": 3581,
        "has_ratings": False, "first_window_empty": False, "extra_separator_ids": (354,),
        "expected_rows": {"train": 151056, "val": 1194, "test": 1194},
        "expected_samples": {"train": 151048, "val": 1184, "test": 1186},
        "train_blocks": 9550, "train_users": 9542,
        "fingerprints": {"train": "d1468b1f36548bdc9c71dca9c918b710695e5efc9052cbe973357d5a10266ee9",
                         "val": "bc63919adc858273ecde739a6eedab8e48bd4359fb204f5179bd55e389d42b75",
                         "test": "987376eb6cb7ac1bc6fdf04b7655bbc467c18277e60bcd305b3717cb89784705"},
        "template_file": "prompt/game.txt", "sasrec_source": "rec_model/steam.pt",
        "title_modes": ("code",), "title_forbidden": ID2NAME_FORBIDDEN,
    },
    "lastfm": {
        "data_dir": "data/ref/lastfm", "item_file": "id2name.txt", "item_format": "id2name", "num_items": 4606,
        "has_ratings": False, "first_window_empty": True, "extra_separator_ids": (),
        "expected_rows": {"train": 50800, "val": 122, "test": 122},
        "expected_samples": {"train": 47872, "val": 122, "test": 122},
        "train_blocks": 976, "train_users": 976,
        "fingerprints": {"train": "890287c1d4f8fc1015d2e7ebe2167e8d098b0a1ff56721a3c73afaecf0cc2640",
                         "val": "9d3440714637ebbf2b19ca9a0f8037fe8a391aa587c75adc736d4d84769bf52c",
                         "test": "13d9b892f7cdf28fc25d61458f3fda873015305cc3486870da9ed24b95a4c354"},
        "template_file": "prompt/artist.txt", "sasrec_source": "rec_model/lastfm.pt",
        "title_modes": ("code",), "title_forbidden": ID2NAME_FORBIDDEN,
    },
}
# §23.2: the other rows of the table (filtered rows, sequences = blocks + val + test rows, user offsets, D21 pool N - 11).
DERIVED = {
    "movielens": ({"train": 2262, "val": 0, "test": 0}, 943, {"train": 0, "val": 754, "test": 848}, 1671),
    "steam": ({"train": 8, "val": 10, "test": 8}, 11938, {"train": 0, "val": 9550, "test": 10744}, 3570),
    "lastfm": ({"train": 2928, "val": 0, "test": 0}, 1220, {"train": 0, "val": 976, "test": 1098}, 4595),
}
# §23.2 hash table (text flag True only for the titles file and the template, CRLF -> LF).
INPUT_HASHES = {
    "movielens": {
        "data/ref/movielens/train_data.df": ("bac4c88b11e6f76d914c2b22aadbb97315a78432bb79f424f7015775053b9546", False),
        "data/ref/movielens/Val_data.df": ("9a950900d0055649e2983ef403d189d5fdc5bd4f84d229a377806d371b32e460", False),
        "data/ref/movielens/Test_data.df": ("d38308a0078d4518dc9eeb619c8d1afb53d830712cc7d36ac51cc47eaafc97d6", False),
        "data/ref/movielens/u.item": ("553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701", True),
        "prompt/movie.txt": ("957fc1f27bb930ca5acbe1b545186ecc04fd93ea2c5307644c706ad44395ecd3", True),
        "rec_model/movielens.pt": ("6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9", False),
    },
    "steam": {
        "data/ref/steam/train_data.df": ("90f8e2ec827427d271cf0b03db45ed5ac3231840fbd022ac57cfae6061ed5b63", False),
        "data/ref/steam/Val_data.df": ("84cd6a73ffea7e64e88e0c01357e01ffc7492fac3870dcccb417939c82fc764b", False),
        "data/ref/steam/Test_data.df": ("2766b9faf90e3531a35207c01fbef2f655af97c21a6ba7d18581bbef2bc6ad59", False),
        "data/ref/steam/id2name.txt": ("dada06a1306c36c91fb1ee3f980df7bee894024b3f52d20832ec2340524d7b7b", True),
        "prompt/game.txt": ("cdc7f7160d5e6beac2c31449540b55525cc069c5aa70277c0ca307a544f4b7b3", True),
        "rec_model/steam.pt": ("752c0cd11a8646adca10025b5746dfcb8968967a6104546f6f869a3ebf26609b", False),
    },
    "lastfm": {
        "data/ref/lastfm/train_data.df": ("7db71a60d3da4c449091fbc254efc9602f81eec6eb9a60534811ccd61974dbc1", False),
        "data/ref/lastfm/Val_data.df": ("92a1ddb65ebdeff999f910b65660eb3d8c7cd85e6a19308d80c429481839f8b3", False),
        "data/ref/lastfm/Test_data.df": ("cac058a2842b2a3ed4d39ed1fa0a742f5034a23c9851a2ffc157b4845d703570", False),
        "data/ref/lastfm/id2name.txt": ("b608bcf14300b1608ccd88f4fe240e6345125201ecdb55157395c19787fe4eaf", True),
        "prompt/artist.txt": ("0b85e0e091281f96460942e177273757a540a7f3b3743b500f53ea37e4288094", True),
        "rec_model/lastfm.pt": ("6f94253543e0ca3b0bbd42b4f6b94f2c97c9fdfdd36c3b5ad3f6b576a806dd86", False),
    },
}


def test_table_covers_every_field():
    """``DatasetSpec`` has exactly the fields of §23.4: a new field needs a row in the table above."""
    names = [f.name for f in dataclasses.fields(DatasetSpec)]
    assert sorted(names) == sorted(["name", *TABLE["movielens"]])


@pytest.mark.parametrize("name", list(TABLE))
def test_spec_values_equal_design_table(name):
    """§23.2 value table for the three datasets; ``MOVIELENS`` keeps the values of data.py (§23.4, D28)."""
    spec = SPECS[name]
    assert spec.name == name
    for field, expected in TABLE[name].items():
        assert getattr(spec, field) == expected, (name, field)
    assert spec.pad_id == TABLE[name]["num_items"]                                    # U main.py:142-147
    assert spec.data_files == tuple(f"{TABLE[name]['data_dir']}/{file}" for file in
                                    ("train_data.df", "Val_data.df", "Test_data.df", TABLE[name]["item_file"]))


@pytest.mark.parametrize("name", list(TABLE))
def test_derived_rows_of_the_table(name):
    """§23.2: filtered rows, sequences = blocks + val + test rows, user id offsets (D14), negative pool size (D21)."""
    dropped, sequences, offsets, pool = DERIVED[name]
    spec = SPECS[name]
    assert {split: spec.expected_rows[split] - spec.expected_samples[split] for split in spec.expected_rows} == dropped
    assert spec.train_blocks + spec.expected_rows["val"] + spec.expected_rows["test"] == sequences
    assert spec.user_offset == offsets
    assert spec.num_items - 11 == pool
    assert spec.train_users <= spec.train_blocks


def test_registry_and_selection():
    """``SPECS`` order is the choice order; ``for_template`` is a closed correspondence (D27, §23.4)."""
    assert list(SPECS) == ["movielens", "steam", "lastfm"]
    assert (SPECS["movielens"], SPECS["steam"], SPECS["lastfm"]) == (MOVIELENS, STEAM, LASTFM)
    field = next(f for f in dataclasses.fields(sections.PromptConfig) if f.name == "template_path")
    assert field.metadata["choices"] == ("prompt/movie.txt", "prompt/game.txt", "prompt/artist.txt")
    assert for_template("prompt/game.txt") is STEAM and for_template("prompt/artist.txt") is LASTFM
    assert for_template("prompt/movie.txt") is MOVIELENS
    cfg = types.SimpleNamespace(prompt=types.SimpleNamespace(template_path="prompt/artist.txt"))
    assert spec_of(cfg) is LASTFM
    for path in ("prompt/read.txt", "prompt/Game.txt", ""):
        with pytest.raises(DatasetError, match="template file"):
            for_template(path)


@pytest.mark.parametrize("path", ["prompt/read.txt", "prompt/Game.txt", "prompt/game.txt ", "",
                                  "data/ref/steam/id2name.txt"])
def test_unknown_template_path_is_a_dataset_error(path):
    """D27, §23.4: the correspondence is closed. Any other path is a ``DatasetError`` (a ``ValueError``) that names
    the path and the known template files, through ``for_template`` and through ``spec_of``."""
    cfg = types.SimpleNamespace(prompt=types.SimpleNamespace(template_path=path))
    for select in (lambda: for_template(path), lambda: spec_of(cfg)):
        with pytest.raises(DatasetError) as info:
            select()
        assert isinstance(info.value, ValueError) and repr(path) in str(info.value)
        assert all(spec.template_file in str(info.value) for spec in SPECS.values())


def _sample(**changes) -> data.Sample:
    """A valid train window of three items; ``changes`` replaces single fields."""
    return data.Sample(**{"split": "train", "row": 7, "user_id": 3, "history": (4, 9, 11), "target": 20, **changes})


def test_sample_num_items_is_not_part_of_the_value():
    """D28, §23.4: ``Sample.num_items`` (default 1682, MovieLens) takes no part in equality, hash or ``repr``, so an
    existing sample, its place in a set and its text are what they were before the field existed."""
    field = next(f for f in dataclasses.fields(data.Sample) if f.name == "num_items")
    assert (field.default, field.compare, field.repr) == (MOVIELENS.num_items, False, False)
    movielens, steam = _sample(), _sample(num_items=STEAM.num_items)
    assert (movielens.num_items, steam.num_items) == (1682, 3581)
    assert movielens == steam and hash(movielens) == hash(steam) and len({movielens, steam}) == 1
    assert repr(movielens) == repr(steam) == "Sample(split='train', row=7, user_id=3, history=(4, 9, 11), target=20)"
    assert movielens != _sample(row=8)                                  # the value fields still count


@pytest.mark.parametrize("bad", [0, -1, True, False, 1682.0, "1682", None, np.int64(1682)],
                         ids=["zero", "negative", "true", "false", "float", "str", "none", "numpy_int"])
def test_bad_num_items_is_a_data_error(bad):
    """``Sample.__post_init__`` (D28): ``num_items`` is a Python int >= 1 (bool excluded, like every id of §3.1);
    anything else is a ``DataError`` naming the sample."""
    with pytest.raises(data.DataError, match=r"sample 'train' row 7: num_items must be an int >= 1"):
        _sample(num_items=bad)


def test_movielens_aliases():
    """The MovieLens module constants stay the same names and are ``MOVIELENS`` values (§23.4, 23.7)."""
    assert data.NUM_ITEMS == data.PAD_ID == batching.PAD_ITEM == MOVIELENS.num_items == 1682
    assert data.DATA_DIR == MOVIELENS.data_dir and data.ITEM_FILE == MOVIELENS.item_file
    assert data.USER_OFFSET == MOVIELENS.user_offset and data.TRAIN_USERS == MOVIELENS.train_users
    assert data.EXPECTED_ROWS == MOVIELENS.expected_rows and data.EXPECTED_SAMPLES == MOVIELENS.expected_samples
    assert data.EXPECTED_FINGERPRINTS == MOVIELENS.fingerprints
    assert data.SPLIT_FILES == {"train": "train_data.df", "val": "Val_data.df", "test": "Test_data.df"}
    assert cluster.DATA_FILES == MOVIELENS.data_files
    assert evaluate.FULL_SPLIT_SIZE == {"val": 94, "test": 95}
    assert prompts.TEMPLATE_FILE == MOVIELENS.template_file
    assert prompts._TITLE_FORBIDDEN == MOVIELENS.title_forbidden


def test_id2name_title_rules():
    """D39: Steam and LastFM forbid exactly the slot strings of M4 and M5 and the special token strings."""
    expected = {prompts.HISTORY_SLOT, prompts.CANDS_SLOT, prompts.HISTORY_TAG.strip(), prompts.CANS_TAG.strip(),
                prompts.PH_TAG.strip(), *batching.EXTRA_TOKENS, batching.PAD_TOKEN, *prompts._SPECIAL_STRINGS}
    assert set(STEAM.title_forbidden) == set(LASTFM.title_forbidden) == expected
    assert "[" not in STEAM.title_forbidden and "]" not in LASTFM.title_forbidden      # titles with one bracket stay


@pytest.mark.parametrize("name", list(TABLE))
def test_expected_inputs_table(name):
    """§23.2 hash table: six entries per dataset, keys = the four data files, template and SASRec source (D28)."""
    table = run.expected_inputs(name)
    assert table == INPUT_HASHES[name]
    spec = SPECS[name]
    assert list(table) == [*spec.data_files, spec.template_file, spec.sasrec_source]
    text = {rel for rel, (_, is_text) in table.items() if is_text}
    assert text == {f"{spec.data_dir}/{spec.item_file}", spec.template_file}


@pytest.mark.parametrize("name", list(TABLE))
def test_expected_inputs_match_the_files_on_disk(name):
    """§23.2, D33: every expected hash is the hash of the file in the working tree (text files after CRLF -> LF)."""
    for rel, (sha, is_text) in run.expected_inputs(name).items():
        assert run.content_sha256(REPO_ROOT / rel, is_text) == sha, rel
    assert run.input_hashes(REPO_ROOT, name) == {rel: sha for rel, (sha, _) in INPUT_HASHES[name].items()}


def test_input_hashes_default_is_movielens():
    """``input_hashes()`` without a dataset keeps its keys: ``start_meta`` and the old tests are unchanged (§23.4)."""
    assert list(run.input_hashes(REPO_ROOT)) == list(INPUT_HASHES["movielens"])
    assert run.expected_inputs() is run.EXPECTED_INPUTS


def test_expected_inputs_unknown_dataset():
    """Only the three table names exist (§23.4)."""
    with pytest.raises(ValueError, match="no expected input table for dataset 'goodreads'"):
        run.expected_inputs("goodreads")


def test_expected_inputs_reads_the_module_attribute(monkeypatch):
    """``expected_inputs`` is the run-time accessor: a patched table is what ``input_hashes`` sees (§23.4)."""
    for name, attr in (("movielens", "EXPECTED_INPUTS"), ("steam", "EXPECTED_INPUTS_STEAM"),
                       ("lastfm", "EXPECTED_INPUTS_LASTFM")):
        table = {"prompt/read.txt": ("0" * 64, True)}
        monkeypatch.setattr(run, attr, table)
        assert run.expected_inputs(name) is table
        assert list(run.input_hashes(REPO_ROOT, name)) == ["prompt/read.txt"]


def test_check_environment_uses_the_dataset_table(monkeypatch):
    """``check_environment(dataset=)`` compares the inputs of that dataset (§23.4)."""
    monkeypatch.setattr(run, "EXPECTED_INPUTS_LASTFM", {"prompt/artist.txt": ("0" * 64, True)})
    with pytest.raises(run.EnvironmentCheckError, match="hash mismatch prompt/artist.txt"):
        run.check_environment(require_gpu=False, llm_path=None, repo_root=REPO_ROOT, dataset="lastfm")


def test_config_checks_read_import_time_tables(monkeypatch):
    """§23.4: the tables of ``sections.py`` are bound at import time and equal ``run.expected_inputs``, so the
    patch of tests/test_m0_config_run.py:632 cannot make ``load_config`` fail (multi/critic/c10_patch.py)."""
    assert set(sections._INPUT_TABLES) == set(SPECS)
    for name in SPECS:
        assert sections._INPUT_TABLES[name] is run.expected_inputs(name)
    full_steam = sections._INPUT_TABLES["steam"]
    monkeypatch.setattr(run, "EXPECTED_INPUTS", {"prompt/movie.txt": ("0" * 64, True)})
    monkeypatch.setattr(run, "EXPECTED_INPUTS_STEAM", {"prompt/game.txt": ("0" * 64, True)})
    assert load_config(CONFIG_DIR / "smoke_cpu.yaml")[0].sasrec.source_path == "rec_model/movielens.pt"
    assert load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml")[0].sasrec.source_path == "rec_model/steam.pt"
    assert sections._INPUT_TABLES["steam"] is full_steam and len(full_steam) == 6


# §23.7 (a): config_hash of multi/rebuild_parametrize/golden_profiles.txt and the sha256 of dump_config's output with
# CRLF replaced by LF (golden_profiles.txt holds the Windows CRLF bytes; config_hash and this value are the same on
# Windows and on VESSL).
GOLDEN = {
    "cluster_hard": ("acb19aedff776114069c3fbaa551c3d0e057c7079441217302d0f075c4b2738a",
                     "54b19990510b43d4b83d99613064f97f9dcd17a20f8ce1c77b272f5d38cc1286"),
    "llara_baseline": ("7d55841fd8f1857f177cdda55d5f45cedce7ad4083b2877f4b7aa4ed41ee9195",
                       "24edf7d01bdcd1fb1689468e440368a068d6b1ede8eba9d9b6bf441fd20f4166"),
    "overfit_gpu": ("7d99dc3f4601e83fad8faa1dd25a0c28dd6cc3384500d3fcf3b05d01e43274b5",
                    "a4f3439eb4e7dabfee1707225a9c7d9601d6e35d8b05f802ca928824cc889b17"),
    "paper_spec": ("2d8ba1dad9527c3b21ac9ab42ded9f46f971778e968871bff7b06c19e1ebfd17",
                   "1a07cb4e2781c94f78b71801e5c12be1612aeb2b263e2de7d2bd8cf90a9ae502"),
    "smoke_cpu": ("269ae8dc07e3f23e1e0cc8a8eef8c2414295a4eb1c9dafc47a204687df9a57fc",
                  "7563eabb878f3be616e987db73d6deb9a92fd3e7395b8e4a8790570ee0405ea7"),
    "smoke_gpu_100": ("a016cfb108a15b6edc20fe6181437adddd4c4c31e968c84454b61ae55306b49d",
                      "98ae5c93e05853283e5ed5d82cbe0db62c3d1a8a158bcb64728ab201936dd45b"),
    "variant_all_hybrid": ("3f5a47b94f0f5c80e90b812fdc200ba9840358bcc0bb8baa36790923e326a3bd",
                           "9a3146e36e8e42a8c015d884e2f9144ef7c9aaee427290add7be11f050cae8d9"),
    "variant_bf16": ("ebec899f647ec589b4fc1684ebb564291f6a15487cc5447e7387745e03e94326",
                     "40d5245298757b8b076c329ed7975e0cf775ed6b0bac19b9f29f1af55fb90c11"),
    "variant_fixed_template": ("40d6aae7c60ecc6b834dfb5f689aff9946b92727301eda9f832bd5a32099da7f",
                               "70e48b9a713ab4620d48fc3841409b21dce377da335384e1b6e404c5e3e8008f"),
    "variant_fixed_titles": ("e0914a5a1b782fd4d65a608835b200c756d49da2000f0b762e33e7b60c873dfd",
                             "d62c8fa51d7df7c3154a63b4e9d41dcd23d057fe348c6149327fbf26a42526af"),
    "variant_released_gate_lr": ("3663bb3919d798923981685bdfb78ea92a7fd1472f5bced84c4ce1487b7cb424",
                                 "ebef2a98b9b6a8a5899c2eb69a94c64c472cb698d439e88aac8572c090846f67"),
    "variant_released_label": ("5e96b7823e7f078421c00edf9f64120be6f87c679b673d00225a466ddf3eccf0",
                               "cc63fdb6028de8ca29a221bc4ba11b0a79d26e97e111171d983f23ae498e8039"),
    "variant_released_lr": ("7769607221ac4e7f4c73873ea363b2bebb3408fca1f684e84f3819b8c0164550",
                            "8bdd8b345814780be5867e740dacf85efb6ba7d38e2c07b199ef72e0aa256297"),
    "variant_reverse_curriculum": ("83a96f48577cabeda5fbba5672c6df2307de761ff3d2d3c4a751fd1985c95c07",
                                   "d34f7399ce892234923ea39dac1d4091eab3c9e1b52052287c1ecb4541e9c0fd"),
    "variant_text_only_eval": ("e0b7108ddd4071a9b9ddddf2deadf22570a0f7f2b6fefa34d4865b09e6ece161",
                               "8528015933d3e1a8ed57977688c571b3945b6e4bdbe5907ebd85b3c688027b44"),
    "variant_total_rank_scaling": ("06dbbf3c09046be3c7225c2196238cb33c04f31ba10c1fdc3868d79f43c77eb7",
                                   "0b267a4e8840c20a942d9b3f6541701143c0550a1511076b71601cb85aef1338"),
    "variant_user_level_cluster": ("9c270e554f71eb7d114146ed8d28555a71d090a9eae08406237fa5a357576a3d",
                                   "9234e007d7b23746560e1a1c47ed3851ed5a1be776c93cf7b531d1fba75b5102"),
}


def test_golden_lists_every_top_level_profile():
    """The 17 MovieLens profiles of ``configs/*.yaml`` (the ``datasets/`` subfolder is not one of them, D43)."""
    assert sorted(GOLDEN) == sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) and len(GOLDEN) == 17


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_movielens_profile_is_unchanged(name, tmp_path):
    """§23.7 (a): ``config_hash`` and the dumped ``config.yaml`` of every MovieLens profile stay what they were
    at commit 54e5ae0, and the profile selects MovieLens through ``prompt.template_path`` (D27)."""
    expected_hash, expected_dump = GOLDEN[name]
    cfg, _ = load_config(CONFIG_DIR / f"{name}.yaml", [ARTIFACT_SET] if name in NEEDS_ARTIFACT else [])
    assert spec_of(cfg) is MOVIELENS
    assert config_hash(cfg) == expected_hash
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected_dump
