from rapids.js import create_node, create_proxy, log
from rapids.util import print

class SignalFactory:

    def __init__(self, type, origin):
        self.type = type
        self.origin = origin

    async def emit(self, value):
        await self.origin.emit(Signal(self.type, self.origin, value))


class Signal:

    def __init__(self, type, origin, value):
        self.type = type
        self.origin = origin
        self.value = value


def find_descriptor(instance, attrname):
    '''Find the descriptor handling a given attribute, if any.

    If the attribute named attrname of the given instance is handled by a
    descriptor, this will return the descriptor object handling the attribute.
    Otherwise, it will return None.
    '''
    def hasspecialmethod(obj, name):
        return any(name in klass.__dict__ for klass in type(obj).__mro__)
    for klass in type(instance).__mro__:
        if attrname in klass.__dict__:
            descriptor = klass.__dict__[attrname]
            if not (hasspecialmethod(descriptor, '__get__') or
                    hasspecialmethod(descriptor, '__set__') or
                    hasspecialmethod(descriptor, '__delete__')):
                # Attribute isn't a descriptor
                return None
            if (attrname in instance.__dict__ and
                not hasspecialmethod(descriptor, '__set__') and
                not hasspecialmethod(descriptor, '__delete__')):
                return None
            return descriptor
    return None


class DomAttribute:
    """
    Attributes that are stored on the DOM
    """

    def __init__(self, name=None, default=None, set_attribute=True):
        self._name = None
        self._dom_name = name
        self._default = default
        self._set_attribute = set_attribute

    def __set_name__(self, owner, name):
        self._name = name
        if self._dom_name is None:
            self._dom_name = name

    def __get__(self, obj, objtype=None):
        if obj.dom == None:
            return self._default
        if self._set_attribute:
            return obj.dom.getAttribute(self._dom_name)
        else:
            return getattr(obj.dom, self._dom_name)

    def __set__(self, obj, value):
        if obj.dom == None:
            obj._lazy_attributes[self._name] = value
        else:
            if self._set_attribute:
                obj.dom.setAttribute(self._dom_name, value)
            else:
                setattr(obj.dom, self._dom_name, value)


class StyleAttribute:
    """
    Attributes that are stored as style on the DOM
    """

    def __init__(self, name=None, default=None):
        self._name = None
        self._dom_name = name
        self._default = default

    def __set_name__(self, owner, name):
        self._name = name
        if self._dom_name is None:
            self._dom_name = name

    def __get__(self, obj, objtype=None):
        if obj.dom == None:
            return self._default
        return getattr(obj.dom.style, self._dom_name)

    def __set__(self, obj, value):
        if obj.dom == None:
            obj._lazy_attributes[self._name] = value
        else:
            setattr(self.dom.style, self._dom_name, value)


class MetaWidget(type):
    """
    Automatically adds DomAttribute descriptors for every attribute in the `dom_attributes` attribute.
    """

    def __new__(cls, name, bases, classdict):

        if 'dom_attributes' in classdict:
            dom_attributes = classdict['dom_attributes']
            for attr in dom_attributes:
                classdict[attr] = DomAttribute()
        else:
            classdict['dom_attributes'] = []

        for name, attribute in classdict.items():
            if isinstance(attribute, DomAttribute):
                classdict['dom_attributes'].append(name)

        if 'style_attributes' in classdict:
            style_attributes = classdict['style_attributes']
            for attr in style_attributes:
                classdict[attr] = StyleAttribute()
        else:
            classdict['style_attributes'] = []

        for name, attribute in classdict.items():
            if isinstance(attribute, StyleAttribute):
                classdict['style_attributes'].append(name)

        result = type.__new__(cls, name, bases, classdict)
        return result


class Widget(metaclass=MetaWidget):

    dom_node = 'div'
    events = None
    dom_attributes = []
    style_attributes = []
    margin_top = StyleAttribute('margin-top')
    margin_bottom = StyleAttribute('margin-bottom')
    margin_right = StyleAttribute('margin-right')
    margin_left = StyleAttribute('margin-left')

    def __init__(self, children=None, id_children=None, data=None, **kwargs):
        self._lazy_attributes = {}
        if children is None:
            children = []
        if id_children is None:
            id_children = {}
        if data is None:
            data = {}

        self._style = {}
        self.parent = None
        self.children = children
        self.id_children = id_children
        for child in self.id_children.values():
            self.children.append(child)

        self.data = data

        self.dom = None
        self._handlers = {}
        self._js_handlers = {}
        if self.events:
            for event in self.events:
                self.on_js(event, SignalFactory(event, self).emit)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.init()

    def init(self):
        pass

    def set_style(self, **kwargs):
        for key, value in kwargs.items():
            self._style[key] = value
        if self.dom:
            for key, value in kwargs.items():
                setattr(self.dom.style, key, value)


    def on_js(self, event, handler):
        handler = create_proxy(handler)
        l = self._js_handlers.get(event, [])
        self._js_handlers[event] = l
        if handler not in l:
            l.append(handler)

        if self.dom:
            self.dom.addEventListener(event, handler)

    def on(self, event, handler, **kwargs):
        l = self._handlers.get(event, [])
        self._handlers[event] = l
        if (handler, kwargs) not in l:
            l.append((handler, kwargs))

    async def emit(self, signal):
        for handler, kwargs in self._handlers.get(signal.type, []):
            if await handler(signal, **kwargs):
                break
        else:
            if self.parent:
                await self.parent.emit(signal)

    def set_widget(self, id, widget):
        prev_idx = None
        if id in self.id_children:
            prev_idx = self.children.index(self.id_children[id])
            self.remove_widget(self.id_children[id])
        self.id_children[id] = widget

        if prev_idx:
            self.insert_widget(prev_idx, widget)
        else:
            self.add_widget(widget)

    def add_widget(self, widget):
        widget.parent = self
        self.children.append(widget)

        if self.dom:
            child_dom = widget._dom_render()
            self.dom.appendChild(child_dom)

    def insert_widget(self, i, widget):
        widget.parent = self
        self.children.insert(i, widget)

        if self.dom:
            child_dom = widget._dom_render()
            if len(self.children) > i+1:
                ref = self.children[i+1]
                self.dom.insertBefore(ref, child_dom)
            else:
                self.dom.appendChild(child_dom)

    def remove_widget(self, widget):
        id = None
        if not isinstance(widget, Widget):
            id = widget
            if widget in self.id_children:
                widget = self.id_children[widget]
            else:
                return False

        if widget not in self.children:
            return False
        if self.dom:
            child_dom = widget.dom
            self.dom.removeChild(child_dom)
        self.children.remove(widget)
        if id:
            del self.id_children[id]
        return True

    def _dom_new(self):
        dom = create_node(self.dom_node)
        self._dom_init(dom)
        return dom

    def _dom_init(self, dom):
        pass

    def _dom_render(self):
        if self.dom is None:
            dom = self._dom_new()

            for attribute_name in self.dom_attributes:
                descriptor = find_descriptor(self, attribute_name)
                dom_name = descriptor._dom_name
                if attribute_name in self._lazy_attributes:
                    value = self._lazy_attributes[attribute_name]
                else:
                    value = descriptor._default
                if value != None:
                    if descriptor._set_attribute:
                        dom.setAttribute(dom_name, value)
                    else:
                        setattr(dom, dom_name, value)

            for attribute_name in self.style_attributes:
                descriptor = find_descriptor(self, attribute_name)
                dom_name = descriptor._dom_name
                if attribute_name in self._lazy_attributes:
                    value = self._lazy_attributes[attribute_name]
                else:
                    value = descriptor._default
                if value != None:
                    setattr(dom.style, dom_name, value)

            for event in self._js_handlers:
                for handler in self._js_handlers[event]:
                    dom.addEventListener(event, handler)

            for child in self.children:
                dom.appendChild(child._dom_render())

            self.dom = dom
        return self.dom


class Div(Widget):
    text = DomAttribute('textContent', set_attribute=False)


class Button(Widget):
    dom_node = 'button'
    events = ['click']
    label = DomAttribute('textContent', set_attribute=False)


class Raw(Widget):
    html = DomAttribute('innerHTML', set_attribute=False)


class Range(Widget):
    dom_attributes = ['min', 'max', 'step']
    events = ['change']


class Grid(Widget):
    display = StyleAttribute(default='grid')
    columns = StyleAttribute('grid-template-columns', 'repeat(12, 8.3% [col-start])')
    #rows = StyleAttribute('grid-template-rows', 'repeat(12, 8.3%, [row-start])')


class Flexbox(Widget):
    display = StyleAttribute('display', 'flex')
    direction = StyleAttribute('flex-direction', 'row')
    wrap = StyleAttribute('flex-wrap', 'nowrap')


class TextInput(Widget):
    dom_node = 'input'
    type = DomAttribute(default='text')
    value = DomAttribute(set_attribute=False)
