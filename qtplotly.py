from qtpy import QtCore, QtWebEngineCore, QtWebEngineWidgets

import plotly.offline as po
import plotly.graph_objs as go

class PlotlySchemeHandler(QtWebEngineCore.QWebEngineUrlSchemeHandler):
    def __init__(self, app):
        super().__init__(app)
        self.m_app = app

    def requestStarted(self, request):
        url = request.requestUrl()
        name = url.host()
        if self.m_app.verify_name(name):
            fig = self.m_app.fig_by_name(name)
            if isinstance(fig, go.Figure):
                raw_html = '<html><head><meta charset="utf-8" />'
                raw_html += '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>'
                raw_html += "<body>"
                raw_html += po.plot(fig, include_plotlyjs=False, output_type="div")
                raw_html += "</body></html>"
                buf = QtCore.QBuffer(parent=self)
                request.destroyed.connect(buf.deleteLater)
                buf.open(QtCore.QIODevice.WriteOnly)
                buf.write(raw_html.encode())
                buf.seek(0)
                buf.close()
                request.reply(b"text/html", buf)
                return
        request.fail(QtWebEngineCore.QWebEngineUrlRequestJob.UrlNotFound)


class PlotlyApplication(QtCore.QObject):
    scheme = b"plotly"

    def __init__(self, parent=None):
        super().__init__(parent)
        scheme = QtWebEngineCore.QWebEngineUrlScheme(PlotlyApplication.scheme)
        QtWebEngineCore.QWebEngineUrlScheme.registerScheme(scheme)
        self.m_functions = dict()

    def init_handler(self, profile=None):
        if profile is None:
            profile = QtWebEngineWidgets.QWebEngineProfile.defaultProfile()
        handler = profile.urlSchemeHandler(PlotlyApplication.scheme)
        if handler is not None:
            profile.removeUrlSchemeHandler(handler)

        self.m_handler = PlotlySchemeHandler(self)
        profile.installUrlSchemeHandler(PlotlyApplication.scheme, self.m_handler)

    def verify_name(self, name):
        return name in self.m_functions

    def fig_by_name(self, name):
        return self.m_functions.get(name, lambda: None)()

    def register(self, name):
        def decorator(f):
            self.m_functions[name] = f
            return f

        return decorator

    def create_url(self, name):
        url = QtCore.QUrl()
        url.setScheme(PlotlyApplication.scheme.decode())
        url.setHost(name)
        return url