"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    SHARDY_COMMIT = "ba26ed9adb0e1be9be815e4274f9745e285b01a8"
    SHARDY_SHA256 = "8f410ea7c2211c2de6370d1cd4b3fbd29aca0e4544e71eed24eab12aae05dffb"
    # LINT.ThenChange(Google-internal path)

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
    )
