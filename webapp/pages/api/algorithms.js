// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default function handler(req, res) {
  const algorithms = [
    // Filtering methods
    {code: 'obw', name: 'Overlap Blocker', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'sbw', name: 'Standard Blocking', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'qgb', name: 'Q-Grams Blocking', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'eqb', name: 'Extended Q-Grams Blocking', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'sab', name: 'Suffix Arrays Blocking', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'esab', name: 'Extended Suffix Arrays Blocking', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'rba', name: 'Rule-Based Blocker', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'ejoin', name: 'ε-Join', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'knnjoin', name: 'top kNN-Join', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'minhash', name: 'MinHash LSH', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'hlsh', name: 'Hyperplane LSH', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'ctlsh', name: 'Cross-polytope LSH', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'faiss', name: 'FAISS', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'scann', name: 'SCANN', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'deepblocker', name: 'DeepBlocker', scenarios: ['filter', 'progress'], params: ['recall']},
    {code: 'sudowoodo', name: 'Sudowoodo', scenarios: ['filter', 'progress'], params: ['recall']},

    // Verification methods
    {code: 'magellan', name: 'Magellan', scenarios: ['verify'], params: ['recall']},
    {code: 'zeroer', name: 'ZeroER', scenarios: ['verify'], params: ['recall']},
    {code: 'deeper', name: 'DeepER', scenarios: ['verify'], params: ['recall', 'epochs']},
    {code: 'deepmatcher', name: 'DeepMatcher', scenarios: ['verify'], params: ['recall', 'epochs']},
    {code: 'emtransformer', name: 'EMTransformer', scenarios: ['verify'], params: ['recall', 'epochs']},
    {code: 'gnem', name: 'GNEM', scenarios: ['verify'], params: ['recall', 'epochs']},
    {code: 'ditto', name: 'DITTO', scenarios: ['verify'], params: ['recall', 'epochs']},
    {code: 'hiermatcher', name: 'HierMatcher', scenarios: ['verify'], params: ['recall', 'epochs']},
  ];

  res.status(200).json(algorithms);
}
