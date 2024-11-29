module.exports = {
  reactStrictMode: true,
  experimental: {
    swcPlugins: [
      [
        'next-superjson-plugin', // without this, Date is not serialized correctly
        {
          excluded: [],
        },
      ],
    ],
  },
}
