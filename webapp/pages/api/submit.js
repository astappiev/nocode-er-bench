// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

export default function handler(req, res) {
  const results = {
    'deepmatcher': {
      'D1': {
        'f1': 98.65,
      },
    },
    'ditto': {
      'D1': {
        'f1': 51.46,
      }
    },
    'emtransformer': {
      'D1': {
        'f1': 98.99,
      }
    },
    'gnem': {
      'D1': {
        'f1': 98.21,
      }
    },
    'hiermatcher': {
      'D1': {
        'f1': 98.21,
      }
    },
    'magellan': {
      'D1': {
        'f1': 97.65,
      }
    },
    'zeroer': {
      'D1': {
        'f1': 98.80,
      }
    },
  };

  if (req.method === 'POST') {
    if (!req.body.dataset || !req.body.algorithm) {
      res.status(400)
    }

    let theResults = results[req.body.algorithm][req.body.dataset];

    if (!theResults) {
      theResults = {
        'f1': 50.00,
      }
    }

    res.status(200).json(Object.entries(theResults).map(([name, value]) => ({name, value})))
  } else {
    res.status(405)
  }
}
