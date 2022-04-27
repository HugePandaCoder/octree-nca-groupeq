import pdoc
import os

"""https://pdoc.dev/docs/pdoc.html"""

def addFileToDocumentation(path):
    os.system('pdoc --html ' + path + ' --force')

def main():
    os.system('pdoc src ') #--force --html
    exit()
    modules = ['Experiment', '/src/agents/Agent']  # Public submodules are auto-imported
    context = pdoc.Context()

    modules = [pdoc.Module(mod, context=context)
            for mod in modules]
    pdoc.link_inheritance(context)

    def recursive_htmls(mod):
        yield mod.name, mod.html()
        for submod in mod.submodules():
            yield from recursive_htmls(submod)

    #for mod in modules:
    #    for module_name, html in recursive_htmls(mod):
    #        ...  # Process

if __name__ == '__main__':
    main()

