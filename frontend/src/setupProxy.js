const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Proxy API requests to the backend server
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
      logLevel: 'debug',
      onError: (err, req, res) => {
        console.error('Proxy error:', err);
        res.status(500).json({
          error: 'Backend server not available',
          message: 'Make sure the backend server is running on port 5000'
        });
      }
    })
  );

  // Add response headers to suppress hot update requests
  app.use((req, res, next) => {
    if (req.url.includes('.hot-update.')) {
      return res.send(''); // Return empty response for hot-update requests
    }
    next();
  });
};