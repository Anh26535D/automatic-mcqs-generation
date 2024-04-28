import os
import subprocess
from platform import architecture, system

class Annotator:

    def __init__(
            self, 
            senna_dir: str = None, 
            stp_dir: str = None,
            dep_model: str ='edu.stanford.nlp.trees.EnglishGrammaticalStructure',
            verbose: bool = False
        ):
        """
			Args:
            -----
            senna_dir: string
				path of senna directory
			stp_dir: string
				path of stanford parser directory
			dep_model: string
				dependency model
			raise_e: bool
				whether to raise exception or not
	
			Returns:
			--------
			None
        """
        self.senna_path = ''
        self.dep_par_path = ''

        if not senna_dir:
            raise 'Please provide the path of senna directory'
        elif senna_dir.startswith('.'):
            self.senna_path = os.path.realpath(senna_dir) + os.path.sep
        else:
            self.senna_path = senna_dir.strip().rstrip(os.path.sep) + os.path.sep

        if not stp_dir:
            raise 'Please provide the path of stanford parser jar file directory'
        elif not self.check_stp_jar(stp_dir):
            raise 'Stanford parser jar file not found in the given directory'
        else:
            self.dep_par_path = stp_dir + os.path.sep
            
        self.dep_par_model = dep_model
        self.default_jar_cli = ['java', '-cp', 'stanford-parser.jar',
                                self.dep_par_model,
                                '-treeFile', 'in.parse', '-collapsed']
        if verbose:
            print('*' * 100)
            print('Setting up the environment for NLP tools')
            print('Senna path:', self.senna_path)
            print('Dependency parser:', self.dep_par_path)
            print('Stanford parser clr', ' '.join(self.default_jar_cli))
            print('*' * 100)

    def check_stp_jar(self, path):
        """Check if the stanford parser is present in the given directory"""
        files = os.listdir(path)
        for file in files:
            if file.startswith('stanford-parser') and file.endswith('.jar'):
                return True
        return False

    def get_senna_bin(self, os_name):
        """Get the current os executable binary file based on the os name"""
        if os_name == 'Linux':
            bits = architecture()[0]
            if bits == '64bit':
                executable = 'senna-linux64'
            elif bits == '32bit':
                executable = 'senna-linux32'
            else:
                executable = 'senna'
        elif os_name == 'Darwin':
            executable = 'senna-osx'
        elif os_name == 'Windows':
            executable = 'senna-win32.exe'
        return self.senna_path + executable

    def get_senna_tag_batch(self, sentences):
        """
        Communicates with senna through lower level communiction(sub process)
        and converted the console output(default is file writing).
        On batch processing each end is add with new line.

        :param list sentences: list of sentences for batch processes
        :rtype: str
        """
        input_data = ""
        for sentence in sentences:
            input_data += sentence + "\n"
        input_data = input_data[:-1]
        package_directory = os.path.dirname(self.senna_path)
        os_name = system()
        executable = self.get_senna_bin(os_name)
        senna_executable = os.path.join(package_directory, executable)
        cwd = os.getcwd()
        os.chdir(package_directory)
        pipe = subprocess.Popen(senna_executable,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=True)
        senna_stdout = pipe.communicate(input=input_data.encode('utf-8'))[0]
        os.chdir(cwd)
        return senna_stdout.decode().split("\n\n")[0:-1]

    def get_senna_tag(self, sentence):
        """
        Communicates with senna through lower level communiction(sub process)
        and converted the console output(default is file writing)

        :param str/list listsentences : list of sentences for batch processes
        :return: senna tagged output
        :rtype: str
        """
        input_data = sentence
        package_directory = os.path.dirname(self.senna_path)
    
        os_name = system()
        executable = self.get_senna_bin(os_name)
        senna_executable = os.path.join(package_directory, executable)
        cwd = os.getcwd()
        os.chdir(package_directory)
        pipe = subprocess.Popen(senna_executable,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=True)
        senna_stdout = pipe.communicate(input=" ".join(input_data)
                                        .encode('utf-8'))[0]
        os.chdir(cwd)
        return senna_stdout

    def get_dependency(self, parse):
        """
        Change to the Stanford parser direction and process the works

        :param str parse: parse is the input(tree format)
                  and it is writen in as file

        :return: stanford dependency universal format
        :rtype: str
        """
        package_directory = os.path.dirname(self.dep_par_path)
        cwd = os.getcwd()
        os.chdir(package_directory)

        with open(self.senna_path + os.path.sep + "in.parse",
                  "w", encoding='utf-8') as parsefile:
            parsefile.write(parse)
        pipe = subprocess.Popen(self.default_jar_cli,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)
        pipe.wait()

        stanford_out = pipe.stdout.read()
        os.chdir(cwd)

        return stanford_out.decode("utf-8").strip()

    def get_batch_annotations(self, sentences, dep_parse=True):
        """
        :param list sentences: list of sentences
        :rtype: dict
        """
        annotations = []
        batch_senna_tags = self.get_senna_tag_batch(sentences)
        for senna_tags in batch_senna_tags:
            annotations += [self.get_annoations(senna_tags=senna_tags)]
        if dep_parse:
            syntax_tree = ""
            for annotation in annotations:
                syntax_tree += annotation['syntax_tree']
            dependencies = self.get_dependency(syntax_tree).split("\n\n")
            # print (dependencies)
            if len(annotations) == len(dependencies):
                for dependencie, annotation in zip(dependencies, annotations):
                    annotation["dep_parse"] = dependencie
        return annotations

    def get_annotations(self, sentence='', senna_tags=None, dep_parse=True):
        """
        passing the string to senna and performing aboue given nlp process
        and the returning them in a form of `dict()`

        :param str or list sentence: a sentence or list of
                     sentence for nlp process.
        :param str or list senna_tags:  this values are by
                     SENNA processed string
        :param bool  batch: the change the mode into batch
                     processing process
        :param bool dep_parse: to tell the code and user need
                    to communicate with stanford parser
        :return: the dict() of every out in the process
                    such as ner, dep_parse, srl, verbs etc.
        :rtype: dict
        """
        annotations = {}
        if not senna_tags:
            senna_tags = self.get_senna_tag(sentence).decode()
            senna_tags = [x.strip() for x in senna_tags.split("\n")]
            senna_tags = senna_tags[0:-2]
        else:
            senna_tags = [x.strip() for x in senna_tags.split("\n")]
        no_verbs = len(senna_tags[0].split("\t")) - 6

        words = []
        pos = []
        chunk = []
        ner = []
        verb = []
        srls = []
        syn = []
        for senna_tag in senna_tags:
            senna_tag = senna_tag.split("\t")
            words += [senna_tag[0].strip()]
            pos += [senna_tag[1].strip()]
            chunk += [senna_tag[2].strip()]
            ner += [senna_tag[3].strip()]
            verb += [senna_tag[4].strip()]
            srl = []
            for i in range(5, 5 + no_verbs):
                srl += [senna_tag[i].strip()]
            srls += [tuple(srl)]
            syn += [senna_tag[-1]]
        roles = []
        for j in range(no_verbs):
            role = {}
            i = 0
            temp = ""
            curr_labels = [x[j] for x in srls]
            for curr_label in curr_labels:
                splits = curr_label.split("-")
                if splits[0] == "S":
                    if len(splits) == 2:
                        if splits[1] == "V":
                            role[splits[1]] = words[i]
                        else:
                            if splits[1] in role:
                                role[splits[1]] += " " + words[i]
                            else:
                                role[splits[1]] = words[i]
                    elif len(splits) == 3:
                        if splits[1] + "-" + splits[2] in role:
                            role[splits[1] + "-" + splits[2]] += " " + words[i]
                        else:
                            role[splits[1] + "-" + splits[2]] = words[i]
                elif splits[0] == "B":
                    temp = temp + " " + words[i]
                elif splits[0] == "I":
                    temp = temp + " " + words[i]
                elif splits[0] == "E":
                    temp = temp + " " + words[i]
                    if len(splits) == 2:
                        if splits[1] == "V":
                            role[splits[1]] = temp.strip()
                        else:
                            if splits[1] in role:
                                role[splits[1]] += " " + temp
                                role[splits[1]] = role[splits[1]].strip()
                            else:
                                role[splits[1]] = temp.strip()
                    elif len(splits) == 3:
                        if splits[1] + "-" + splits[2] in role:
                            role[splits[1] + "-" + splits[2]] += " " + temp
                            role[splits[1] + "-" + splits[2]] = role[splits[1] + "-" + splits[2]].strip()
                        else:
                            role[splits[1] + "-" + splits[2]] = temp.strip()
                    temp = ""
                i += 1
            if "V" in role:
                roles += [role]
        annotations['words'] = words
        annotations['pos'] = list(zip(words, pos))
        annotations['ner'] = list(zip(words, ner))
        annotations['srl'] = roles
        annotations['verbs'] = [x for x in verb if x != "-"]
        annotations['chunk'] = list(zip(words, chunk))
        annotations['dep_parse'] = ""
        annotations['syntax_tree'] = ""
        for (word_, syn_, pos_) in zip(words, syn, pos):
            annotations['syntax_tree'] += syn_.replace("*", "(" + pos_ + " " + word_ + ")")

        if dep_parse:
            annotations['dep_parse'] = self.get_dependency(annotations
                                                           ['syntax_tree'])
        return annotations