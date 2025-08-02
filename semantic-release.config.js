module.exports = {
  "branches": [
    "main",
    {
      "name": "develop",
      "prerelease": "beta"
    },
    {
      "name": "next",
      "prerelease": "alpha"
    }
  ],
  "plugins": [
    [
      "@semantic-release/commit-analyzer",
      {
        "preset": "angular",
        "releaseRules": [
          {"type": "docs", "scope": "README", "release": "patch"},
          {"type": "refactor", "release": "patch"},
          {"type": "style", "release": "patch"},
          {"type": "test", "release": "patch"},
          {"type": "build", "release": "patch"},
          {"type": "ci", "release": "patch"},
          {"type": "perf", "release": "patch"}
        ],
        "parserOpts": {
          "noteKeywords": ["BREAKING CHANGE", "BREAKING CHANGES"]
        }
      }
    ],
    [
      "@semantic-release/release-notes-generator",
      {
        "preset": "angular",
        "presetConfig": {
          "types": [
            {"type": "feat", "section": "Features"},
            {"type": "fix", "section": "Bug Fixes"},
            {"type": "perf", "section": "Performance Improvements"},
            {"type": "revert", "section": "Reverts"},
            {"type": "docs", "section": "Documentation", "hidden": false},
            {"type": "style", "section": "Styles", "hidden": false},
            {"type": "refactor", "section": "Code Refactoring", "hidden": false},
            {"type": "test", "section": "Tests", "hidden": false},
            {"type": "build", "section": "Build System", "hidden": false},
            {"type": "ci", "section": "Continuous Integration", "hidden": false}
          ]
        }
      }
    ],
    [
      "@semantic-release/changelog",
      {
        "changelogFile": "CHANGELOG.md",
        "changelogTitle": "# Changelog\n\nAll notable changes to the Video Diffusion Benchmark Suite will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n<!-- CHANGELOG -->"
      }
    ],
    [
      "@semantic-release/npm",
      {
        "npmPublish": false,
        "tarballDir": "dist"
      }
    ],
    [
      "@semantic-release/exec",
      {
        "prepareCmd": "echo 'VERSION=${nextRelease.version}' > .release-version",
        "publishCmd": "docker build -t ghcr.io/danieleschmidt/vid-diffusion-benchmark:${nextRelease.version} . && docker push ghcr.io/danieleschmidt/vid-diffusion-benchmark:${nextRelease.version}"
      }
    ],
    [
      "@semantic-release/github",
      {
        "assets": [
          {
            "path": "dist/*.tar.gz",
            "label": "Distribution package"
          },
          {
            "path": "CHANGELOG.md",
            "label": "Changelog"
          }
        ],
        "assignees": ["danieleschmidt"],
        "addReleases": "bottom",
        "failComment": false,
        "failTitle": false,
        "labels": ["release"],
        "releasedLabels": ["released"],
        "successComment": "🎉 This ${issue.pull_request ? 'PR is included' : 'issue has been resolved'} in version [${nextRelease.version}](${releases.filter(release => release.name)[0].url}) 🎉"
      }
    ],
    [
      "@semantic-release/git",
      {
        "assets": ["CHANGELOG.md", "package.json", "package-lock.json", "pyproject.toml"],
        "message": "chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}"
      }
    ]
  ],
  "repositoryUrl": "https://github.com/danieleschmidt/vid-diffusion-benchmark-suite.git",
  "tagFormat": "v${version}",
  "preset": "angular"
}