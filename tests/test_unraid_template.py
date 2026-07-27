"""Validate the Unraid Community Apps template shipped by the project."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "unraid" / "mediasage.xml"
ICON = ROOT / "unraid" / "mediasage-icon.png"


def test_unraid_template_has_required_metadata() -> None:
    root = ET.parse(TEMPLATE).getroot()

    assert root.tag == "Container"
    assert root.attrib == {"version": "2"}
    for tag in (
        "Name",
        "Repository",
        "Registry",
        "Network",
        "Shell",
        "Privileged",
        "Support",
        "Project",
        "Overview",
        "Category",
        "WebUI",
        "TemplateURL",
        "Icon",
    ):
        assert root.findtext(tag), f"missing required <{tag}> field"

    assert root.findtext("Name") == "MediaSage"
    assert root.findtext("Repository") == "ghcr.io/ecwilsonaz/mediasage:latest"
    assert root.findtext("Shell") == "sh"
    assert root.findtext("Privileged") == "false"
    assert root.findtext("WebUI") == "http://[IP]:[PORT:5765]/"


def test_unraid_template_only_forces_runtime_essentials() -> None:
    root = ET.parse(TEMPLATE).getroot()
    configs = root.findall("Config")

    assert {config.attrib["Type"] for config in configs} == {"Port", "Path"}
    assert {config.attrib["Target"] for config in configs} == {"5765", "/app/data"}
    assert all(config.attrib["Required"] == "true" for config in configs)


def test_unraid_icon_is_square_512_png() -> None:
    data = ICON.read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (512, 512)
