import xml.etree.ElementTree as ET


def xml_to_dict(xml_string: str):
    root = ET.fromstring(xml_string)
    return element_to_dict(root)


def element_to_dict(element: ET.Element):
    result = {}
    if element.attrib:
        result.update(element.attrib)
    for child in element:
        child_data = element_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    if element.text:
        result[element.tag] = element.text.strip()
    return result


if __name__ == "__main__":
    # Example XML string
    xml_string = '''
    <annotations>
        <version>1.1</version>
        <meta>
            <task>
            </task>
            <dumped>2019-02-18 20:02:59.930671+03:00</dumped>
        </meta>
        <track label="pedestrian">
            <box frame="0" keyframe="1" occluded="0" outside="0" xbr="533.0" xtl="465.0" ybr="848.0" ytl="730.0">
                <attribute name="id">0_1_3b</attribute>
                <attribute name="old_id">pedestrian1</attribute>
                <attribute name="look">not-looking</attribute>
                <attribute name="reaction">__undefined__</attribute>
                <attribute name="action">standing</attribute>
                <attribute name="cross">not-crossing</attribute>
                <attribute name="hand_gesture">__undefined__</attribute>
                <attribute name="occlusion">none</attribute>
                <attribute name="nod">__undefined__</attribute>
            </box>
            <box frame="1" keyframe="1" occluded="0" outside="0" xbr="532.0" xtl="463.0" ybr="848.0" ytl="730.0">
                <attribute name="id">0_1_3b</attribute>
                <attribute name="old_id">pedestrian1</attribute>
                <attribute name="look">not-looking</attribute>
                <attribute name="reaction">__undefined__</attribute>
                <attribute name="action">standing</attribute>
                <attribute name="cross">not-crossing</attribute>
                <attribute name="hand_gesture">__undefined__</attribute>
                <attribute name="occlusion">none</attribute>
                <attribute name="nod">__undefined__</attribute>
            </box>
            <box frame="2" keyframe="1" occluded="0" outside="0" xbr="531.0" xtl="461.0" ybr="849.0" ytl="730.0">
                <attribute name="id">0_1_3b</attribute>
                <attribute name="old_id">pedestrian1</attribute>
                <attribute name="look">not-looking</attribute>
                <attribute name="reaction">__undefined__</attribute>
                <attribute name="action">standing</attribute>
                <attribute name="cross">not-crossing</attribute>
                <attribute name="hand_gesture">__undefined__</attribute>
                <attribute name="occlusion">none</attribute>
                <attribute name="nod">__undefined__</attribute>
            </box>
        </track>
    </annotations>
    '''

    xml_string_short = '''
        <annotations>
            <version>1.1</version>
            <meta></meta>
            <track label="pedestrian">
            </track>
            <track label="ped">
            </track>
            <track label="pedestrian">
            </track>
            <track label="ped">
            </track>
        </annotations>
        '''

    # Convert XML to dictionary
    xml_dict = xml_to_dict(xml_string)
    print(xml_dict)
