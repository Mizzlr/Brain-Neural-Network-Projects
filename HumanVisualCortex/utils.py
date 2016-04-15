import time

class Timer:

	def __init__(self, message = "Time Elaspsed: "):
		self._message = message
		self.tic = None
		self.toc = None

	def start(self):
		self.tic = time.time()

	def stop(self):
		self.toc = time.time()

	def message(self):
		return self._message + str(self.toc - self.tic)

	def set_message(self, message):
		self._message = message

