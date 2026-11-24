import json
from pathlib import Path
import tempfile
import unittest

from validate_retain_trace import validate_trace


class TemplateControls(unittest.TestCase):
    def validate(self, medium, required=True):
        kwargs = {'enable_thinking': True}
        if medium is not None:
            kwargs['medium_effort'] = medium
        schema = {'type':'object','properties':{'facts':{'type':'array','items':{'type':'object','properties':{'what':{'type':'string'}},'required':['what']}}},'required':['facts']}
        request = {'event':'request','request_id':'test','body':{'model':'candidate','messages':[],
            'chat_template_kwargs':kwargs,'max_tokens':16384,
            'response_format':{'type':'json_schema','json_schema':{'name':'facts','strict':True,'schema':schema}}}}
        response = {'event':'response','request_id':'test','status':200,'body':{
            'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'facts':[{'what':'User plans a test.'}]}),'reasoning_content':'The source describes a plan.'}}],
            'usage':{'completion_tokens':30}}}
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder)/'trace.ndjson'
            trace.write_text(json.dumps(request)+'\n'+json.dumps(response)+'\n')
            return validate_trace(trace,'candidate',True,16384,True,{'medium_effort':True} if required else None)

    def test_exact_native_control_with_separated_reasoning_passes(self):
        self.assertEqual(self.validate(True)['error_count'],0)

    def test_missing_false_and_integer_controls_are_rejected(self):
        for value in (None,False,1):
            with self.subTest(value=value):
                summary = self.validate(value)
                self.assertTrue(any('wrong required template control' in e for e in summary['errors']))

    def test_legacy_call_without_required_control_stays_valid(self):
        self.assertEqual(self.validate(None,required=False)['error_count'],0)


if __name__ == '__main__':
    unittest.main()
