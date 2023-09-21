from js import document

async def run():
    node = document.createElement('div')
    node.innerHTML = 'Hello World'
    root = document.getElementById('root')
    root.appendChild(node)

