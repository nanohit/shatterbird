"""Static server for the probe page; POST /save/<name> stores a result file under ./out/."""
import http.server
import os
import re
import sys


class H(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        m = re.fullmatch(r"/save/([A-Za-z0-9_.-]+)", self.path)
        if not m:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        os.makedirs("out", exist_ok=True)
        with open(os.path.join("out", m.group(1)), "wb") as f:
            f.write(body)
        self.send_response(200)
        self.end_headers()


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
