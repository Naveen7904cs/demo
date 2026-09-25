app.get('/', (req, res) => {
  res.send('Server is running. Access endpoints via https://');
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
