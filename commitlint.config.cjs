//Copyright (c) 2025 Soares
//
// SPDX-License-Identifier: Apache-2.0

module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // 100 --> match CI; 0 --> disable
    'body-max-line-length': [2, 'always', 100],
    // optional: also keep footers short
    'footer-max-line-length': [2, 'always', 100],
  },
};
