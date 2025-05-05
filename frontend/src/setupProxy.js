module.exports = function(app) {
  // Add response headers to suppress hot update requests
  app.use((req, res, next) => {
    if (req.url.includes('.hot-update.')) {
      return res.send(''); // Return empty response for hot-update requests
    }
    next();
  });
}; 