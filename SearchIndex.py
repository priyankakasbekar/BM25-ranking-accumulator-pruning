import os.path
import string
import sys
import math


"""
To Run:
Bm25 Ranking
Python3.6 SearchIndex.py path_to_corpus_file bm25 5 “query” 

Proximity Ranking
Python3.6 SearchIndex.py path_to_corpus_file proximity 5 “query” 
"""


class InvertedIndex:
    def __init__(self, term_dict, num_docs, query_dict=None):
        self.term_dict = term_dict
        self.num_docs = num_docs
        self.query_dict = query_dict

    def first(self, t):
        if self.term_dict[t] is not None:
            return Current(self.term_dict[t][0].doc_index, self.term_dict[t][0].lstposition[0])
        return Current("inf", "inf")

    def last(self, t):
        if self.term_dict[t] is not None:
            return Current(self.term_dict[t][-1].doc_index, self.term_dict[t][-1].lstposition[-1])
        return Current("inf", "inf")

    def prev(self, t, current):
        """
            find prev occurence of the term t from given current position
            We use two iterations of gallop search to prev position in term posting list
            First gallop - finds doc_id_idx
            Second gallop - finds pos from doc searched by 1st iteration
            :param t - term
            :param current - current position
            :return previous position of t from current as (doc, pos) pair
            """
        if t not in self.term_dict:
            return Current("inf", "inf")
        else:
            # end of corpus -> returns last position in last doc
            if current.doc_id == "inf" and current.pos == "inf":
                return Current(self.term_dict[t][-1].doc_index, self.term_dict[t][-1].lstposition[-1])
            # finds doc_id
            doc_id_idx = galloping_search(self.term_dict[t], current.doc_id, True)

            if doc_id_idx == 'inf':
                return Current("inf", "inf")

            # current is not in searched term list, need to return last position from valid doc; need not find pos
            if self.term_dict[t][doc_id_idx].doc_index > current.doc_id:
                if doc_id_idx-1<0:
                    return Current("inf", "inf")
                else:
                    return Current(self.term_dict[t][doc_id_idx-1].doc_index, self.term_dict[t][doc_id_idx-1].lstposition[-1])

            elif self.term_dict[t][doc_id_idx].doc_index < current.doc_id:
                return Current(self.term_dict[t][doc_id_idx].doc_index,
                               self.term_dict[t][doc_id_idx].lstposition[-1])

            elif self.term_dict[t][doc_id_idx].doc_index == current.doc_id:
            # finds pos id
                pos_idx = galloping_search(self.term_dict[t][doc_id_idx].lstposition, current.pos)
            # term not found - so return current
                if self.term_dict[t][doc_id_idx].lstposition[pos_idx] < current.pos:
                    return Current(self.term_dict[t][doc_id_idx].doc_index,
                                   self.term_dict[t][doc_id_idx].lstposition[pos_idx])
                if pos_idx == 'inf':
                    return Current("inf", "inf")

            # search finds a position before current
                else:
                    # end of position list
                    if pos_idx - 1 < 0:
                        # end of doc list as well
                        if doc_id_idx - 1 < 0:
                            return Current("inf", "inf")
                        else:
                        # if not end of doclist but end of position list, return last element of prev doc
                            return Current(self.term_dict[t][doc_id_idx - 1].doc_index,
                                           self.term_dict[t][doc_id_idx - 1].lstposition[-1])
                    else:
                    # not end of doc list or pos list within it. normal case
                        return Current(self.term_dict[t][doc_id_idx].doc_index, self.term_dict[t][doc_id_idx].lstposition[pos_idx - 1])

    def next(self, t, curr):
        """
        Find next instance of term from given 'current' position.
        We use two iterations of gallop search to next position in term posting list
        First gallop - finds doc_id_idx
        Second gallop - finds pos from doc searched by 1st iteration
        If current is present, galloping search returns it, and we find the next occurance
        If current is not present in positing list, search returns next nearest value and return that directly
        :param t - term
        :param current - current position
        :return next position of t from current as (doc, pos) pair
        """
        current = Current(curr.doc_id, curr.pos)
        if t not in self.term_dict:
            return Current("inf", "inf")
        else:
            # find doc in which current is present
            beginning = False
            if current.doc_id == "-inf":
                beginning = True
                current.doc_id = 0
                current.pos = 0
            doc_id_idx = galloping_search(self.term_dict[t], current.doc_id, True)
            if doc_id_idx == 'inf':
                return Current("inf", "inf")

            # current not found. next nearest is in next doc, so return first pos
            if self.term_dict[t][doc_id_idx].doc_index > current.doc_id:
                return Current(self.term_dict[t][doc_id_idx].doc_index, self.term_dict[t][doc_id_idx].lstposition[0])

            # find pos in given doc from prev gallop
            pos_idx = galloping_search(self.term_dict[t][doc_id_idx].lstposition, current.pos)

            if beginning:
                return Current(self.term_dict[t][doc_id_idx].doc_index,
                               self.term_dict[t][doc_id_idx].lstposition[pos_idx])

            # search finds a position before current
            if self.term_dict[t][doc_id_idx].lstposition[pos_idx] <= current.pos \
                    and self.term_dict[t][doc_id_idx].doc_index == current.doc_id:
                if pos_idx == 'inf':
                    return Current("inf", "inf")
                # position found by search is at the end of posting list
                if pos_idx + 1 == self.term_dict[t][doc_id_idx].termfrequency:
                    # end of corpus
                    if len(self.term_dict[t]) - 1 == doc_id_idx:
                        return Current("inf", "inf")
                    else:
                        # end of position list but not last doc
                        return Current(self.term_dict[t][doc_id_idx + 1].doc_index,
                                       self.term_dict[t][doc_id_idx + 1].lstposition[0])
                else:
                    # same doc and has more items in same doc list - normal case
                    return Current(current.doc_id, self.term_dict[t][doc_id_idx].lstposition[pos_idx + 1])
            elif current.pos < self.term_dict[t][doc_id_idx].lstposition[pos_idx]:
                # not found so default next nearest returned by galloping search is used as next
                return Current(self.term_dict[t][doc_id_idx].doc_index,
                               self.term_dict[t][doc_id_idx].lstposition[pos_idx])


class Current:
    """
    position of term represented as doc_id and position within that doc
    """

    def __init__(self, doc_id, pos):
        self.doc_id = doc_id
        self.pos = pos

    def print(self):
        print("doc_id = {}, pos= {} ", self.doc_id, self.pos)


class Cover:
    """
    Cover represented as pair of (doc, pos) each corresponding to leftmost and rightmost endpoints covering the given terms
    """

    def __init__(self, u_docid="inf", upos="inf", vpos=None):
        if vpos is not None:
            self.u.doc_id = u_docid
            self.v.doc_id = "inf"
            self.u.pos = upos
            self.v.pos = "inf"
        else:
            self.u = u_docid
            self.v = upos

    def printCover(self):
        print("cover :: ")
        print(self.u.doc_id, self.u.pos)
        print(self.v.doc_id, self.v.pos)

    # calculate (v - u + 1)
    def dist(self, posadd):
        return self.v.pos - self.u.pos + posadd


class PositionalIndex:
    """
    Every Position Index object has three members - doc_index, termfrequency, position in the document
    """

    def __init__(self, doc_index, position):
        self.doc_index = doc_index
        self.termfrequency = 1
        self.lstposition = position

    # if the positional index object for the given doc is already present, only the frequency of the term has to
    # be updated
    def incrementDocFreqCount(self, position):
        self.termfrequency += 1
        self.lstposition.append(position)
        return True

    # posting list of every term should end with infinity
    def appendEndOfFile(self):
        self.lstposition.append(float("inf"))

    def printIndex(self):
        print("document index", self.doc_index)
        print("term frequency", self.termfrequency)
        print("lstpositions", self.lstposition)


class Ranking:
    """
    This class is used to store results of ranking mathods (bm25, proximity) as (doc_id, score)
    """

    def __init__(self, doc_id, score):
        self.doc_id = doc_id
        self.score = score


"""
Helper functions
"""


def getFileContents(filename):
    """
    Read the file and return it's contents
    :param filename:
    :return: contents of text file
    """

    if not os.path.exists(filename):
        print("File doesn't exist")
        return

    fileText = open(filename)
    text = fileText.read()
    return text


def writeFileContents(filename, results):
    """
    Write results into the file specified by filename
    :param filename:
    :return:
    """
    if not os.path.exists(filename):
        fileText = open(filename, "w+")
        for result in results:
            fileText.write('0' + ' '+ '0'+ ' '+ 'docnum'+str(result.doc_id) + ' ' + str(results.index(result)) + ' '
                           + str(result.score) + ' '+ 'my_test' + '\n')
        fileText.close()
        return True

    if os.path.getsize(filename) == 0:
        last_query_id = -1
    else:
        f = open(filename, 'r')
        lastline = f.readlines()[-1]
        last_query_id = int(lastline.split()[0])
        f.close()

    fileText = open(filename, "a")

    for result in results:
        fileText.write(str(last_query_id +1) + ' '+ '0'+ ' '+ 'docnum'+str(result.doc_id) + ' ' + str(results.index(result))
                       + ' ' + str(result.score) + ' '+ 'my_test' + '\n')
    fileText.close()
    return True


def createPositionalIndex(corpus):
    """
    For every term in the corpus, a list of positional index objects. Each positional index object contains the
    doc_id in which the term is present, frequency of the term in that document, position of the term in that document
    :param corpus:
    :return: positional index for the corpus
    """
    # entire corpus file is split into multiple documents
    docs = corpus.split('\n\n')
    dict_docs = {}
    num_of_docs = len(docs)
    # term_dict contains the positional index of the entire corpus. It is a dictionary with key as the term, the value
    # is the list of positional index objects
    internal_term_dict = {}

    # iterate over each document
    for doc_idx, doc in enumerate(docs):

        words = doc.split()
        corpus_doc_freq[doc_idx] = len(words)

        # i is used to keep track of position within the document
        i = 0
        updated = False

        # iterate over each word
        for word in words:
            # remove the punctuations
            word_without_punct = Removepunctuation(word)
            # change the case of every term to lower case
            word_without_punct = word_without_punct.lower()
            '''
            if the term is not present in term_dict, create a new positional index object. Fill the position of
            the term beginning with -inf and then the position of first occurrence of the term.
             if the term is present in the dictionary, check if the entry for the current doc_id is already present.
            If present, then just increment the term frequency. Else, create a new positional index object and 
            append it to the list which is the value of the dictionary
            '''
            if word_without_punct not in internal_term_dict:
                internal_term_dict[word_without_punct] = [PositionalIndex(doc_idx, [words.index(word, i)])]
            else:
                for pos_obj in internal_term_dict[word_without_punct]:
                    if pos_obj.doc_index == doc_idx:
                        updated = pos_obj.incrementDocFreqCount(words.index(word, i))
                if not updated:
                    internal_term_dict[word_without_punct].append(PositionalIndex(doc_idx, [words.index(word, i)]))
                updated = False

            i += 1

    return num_of_docs, internal_term_dict


def binary_search(lstposition, low, high, current_value, check_doc_id=False):
    """
    	:param lstposition: List of objects or values over which the galloping search has to be performed.
    	:param low: Lower limit for the binary search.
    	:param high: Higher limit for the binary search.
    	:param current_value: Value which will be searched in the lstposition list.
    	:param check_doc_id: Boolean for enabling flows within the galloping search.
    	:return: Position or index of the current_value within the lstposition.
    	    If the exact current_value is not found, next
    	nearest value's index is returned
    	"""
    # Checking the boolean to determine the flow - Flow for searching on a list of positions within a document index
    if not check_doc_id:
        # calculating the mid using low and high
        mid = int((low + high) / 2)
        if low < high:
            # if the position value at mid index is greater than the current_value, call the binary search with low, mid as high
            if lstposition[int(mid)] > current_value:
                return binary_search(lstposition, low, mid, current_value, check_doc_id)
            # if the position value at mid index is less than the current_value, call the binary search with mid as low, high
            elif lstposition[int(mid)] < current_value:
                return binary_search(lstposition, mid + 1, high, current_value, check_doc_id)
        # return mid if the value at mid position equals the current_value
        return int(mid)
    # Flow for searching the document id on a list of objects containing document id and positions
    else:
        # calculating the mid using low and high
        mid = int((low + high) / 2)
        if low < high:
            # if the document id at mid index is greater than the current_value, call the binary search with low, mid as high
            if lstposition[mid].doc_index > current_value:
                return binary_search(lstposition, low, mid, current_value, check_doc_id)
            # if the position value at mid index is less than the current_value, call the binary search with mid as low, high
            elif lstposition[mid].doc_index < current_value:
                return binary_search(lstposition, mid + 1, high, current_value, check_doc_id)
        return mid
    return mid


def galloping_search(lstposition, current_value, check_doc_id=False):
    """
    :param lstposition: List of objects or values over which the galloping search has to be performed.
    :param current_value: Value which will be searched in the lstposition list.
    :param check_doc_id: Boolean for enabling flows within the galloping search.
    :return: The value that is returned by the binary search.
    Galloping search method performs a general galloping (exponential) search over the given parameters. The search
    works by exponentially increasing the high value. When the value present at the high index position is less than
    the current value, the high value is multiplied by 2. In this case, the high value is set to low before high value
    increases exponentially. When the value present at the high index position is greater than the current_value,
    we fix the high value. After this, the binary search method is called with the high and low values.
    """
    # initialization
    low = 0
    jump = 1
    high = low + jump
    # Flag to determine the flow - for document id search
    if check_doc_id:
        # Iterate until the high value is less than the length of the document id list and
        # value at high is less than the current_value
        while (high < (len(lstposition) - 1) and lstposition[high].doc_index < current_value):
            low = high
            jump = 2 * jump
            high = low + jump
        # If high values is greater than the length of the object list, set high to the length of the object list
        if high > (len(lstposition) - 1):
            high = len(lstposition) - 1

    # Flag to determine the flow - for position index search
    else:
        # Iterate until the high value is less than the length of the position list and value at high is less than the current_value
        while (high < (len(lstposition) - 1) and lstposition[high] < current_value):
            low = high
            jump = 2 * jump
            high = low + jump
        # If high values is greater than the length of the position list, set high to the length of the position list
        if high > (len(lstposition) - 1):
            high = len(lstposition) - 1

    return binary_search(lstposition, low, high, current_value, check_doc_id)


def Removepunctuation(term):
    """
        Removes the punctuations from a term using the translator
    :param term:
    :return:
    """

    translator = str.maketrans(dict.fromkeys(string.punctuation))
    term_no_punct = term.translate(translator)
    return term_no_punct


def nextDoc(term, doc_id):
    """
    :param term: the term whose occurence in the next document needs to be found
    :param doc_id: the current document id in which the term is present
    :param isdoccorpus: This boolean variable decides if we need to use positional index of the document corpus or
                        the positional index of the query corpus
    :return: the positional index of document_id of the document greater than doc_id in which the term appears
    This function uses the galloping method to set the higher and lower bounds for the nextDoc search
    """
    global term_dict

    low = 0
    jump = 1
    high = low + jump

    local_dict = term_dict

    if doc_id == float("-inf"):
        return local_dict[term][0]

    if local_dict[term][len(local_dict[term]) - 1].doc_index <= doc_id:
        return float("inf")

    else:
        while (high < len(local_dict[term]) - 1 and local_dict[term][high].doc_index <= doc_id):
            low = high
            jump = 2 * jump
            high = low + jump

        if high > len(local_dict[term]) - 1:
            high = len(local_dict[term]) - 1

        return binary_search_doc(local_dict[term], doc_id, low, high)


def binary_search_doc(lst_pos_obj, doc_id, low, high):
    """
    :param lst_pos_obj: The lst_pos_obj will contain the list of positional index objects
    :param doc_id: The current doc_id in which the term is present
    :param low: lower bound for the binary search
    :param high: higher bound for the binary search
    :return: Return the position of the positional object that has doc_index > doc_id
    """
    if low == high:
        return lst_pos_obj[low]
    if (high - low) == 1:
        if lst_pos_obj[low].doc_index > doc_id:
            return lst_pos_obj[low]
        elif lst_pos_obj[high].doc_index > doc_id:
            return lst_pos_obj[high]

    while (low < high):
        mid = (low + high) / 2
        mid = int(mid)

        if lst_pos_obj[mid].doc_index < doc_id:
            return binary_search_doc(lst_pos_obj, doc_id, int(mid), high)
        elif lst_pos_obj[mid].doc_index > doc_id:
            return binary_search_doc(lst_pos_obj, doc_id, low, mid)
        elif lst_pos_obj[mid].doc_index == doc_id:
            return lst_pos_obj[mid + 1]

def TFBM25(doc_length,term_freq,doc_avg_length):
    """

    :param doc_length:
    :param term_freq:
    :param doc_avg_length:
    :return: BM25 score
    """
    k = 1.2
    b = 0.75

    return (term_freq * (k + 1))/(term_freq + k*(1-b) + b*(doc_length/doc_avg_length))


def findDocLengths(corpus):
    """
    Create and return a dictionary with document id as key and the number of terms in the document (length) as the value
    :param corpus:
    :return:
    """
    dict_doc_length = {}

    docs = corpus.split('\n\n')

    for (doc_id,doc) in enumerate(docs):
        dict_doc_length[doc_id] = len(doc.split())

    return dict_doc_length


def BM25(corpus,query,num_of_results):
    """
    performs term at a time processing on the query terms and computes the ranking using bm25 as the ranking scheme
    uses accumulator pruning to limit the number of documents ranked. R
    :param corpus:
    :param query:
    :param num_of_results:
    :return: the scores for top k documents
    """

    global term_dict
    (num_of_docs, term_dict) = createPositionalIndex(corpus)
    dict_doc_length = findDocLengths(corpus)
    acc_prev = []
    acc_current = []
    a_max = int(num_of_results) + 2

    #split the query into terms
    query_terms = query.split()
    query_terms = [Removepunctuation(query_term).lower() for query_term in query_terms]

    doc_avg_length = len(term_dict)/num_of_docs

    #create a dictionary with query term as key and the number of documents it appears in as the value
    query_term_freq = {}
    for queryterm in query_terms:
        if queryterm in term_dict:
            query_term_freq[queryterm] = len(term_dict[queryterm])
        else:
            query_term_freq[queryterm] = 0
    # sort the doctionary based on values
    #query_term_sorted = sorted(query_term_freq,key = lambda x:x[1])
    query_term_sorted = [(k, query_term_freq[k]) for k in sorted(query_term_freq, key=query_term_freq.get, reverse=True)]
    query_terms = []

    for (term,freq) in query_term_sorted:
        if freq != 0:
            query_terms.append(term)

    acc_prev.append(Ranking(float("inf"),0))

    for query_term in query_terms:
        quota_left = a_max - len(acc_prev)
        inpos = 0
        outpos = 0
        # if all the postings of the term can fit within the available quota
        if query_term_freq[query_term] <= quota_left:
            pos_obj = nextDoc(query_term,float("-inf"))
            doc_id = pos_obj.doc_index
            while doc_id < float("inf"):
                # copy the document scores of the doc_ids before the current doc_id
                while acc_prev[inpos].doc_id < doc_id:
                    if len(acc_current) >= inpos + 1:
                        acc_current[outpos].doc_id = acc_prev[inpos].doc_id
                        acc_current[outpos].score = acc_prev[inpos].score
                    else:
                        acc_current.append(
                            Ranking(doc_id, (math.log2(num_of_docs / query_term_freq[query_term])
                                    * TFBM25(dict_doc_length[doc_id], pos_obj.termfrequency, doc_avg_length))))
                    outpos += 1
                    inpos += 1
                #acc_current.append(Ranking(doc_id,math.log2(num_of_docs/query_term_freq[query_term]) *
                                           #TFBM25(dict_doc_length[doc_id],pos_obj.termfrequency,doc_avg_length)))

                # if the previous terms were also present in the current document, add up the scores from previous documents
                if acc_prev[inpos].doc_id == doc_id:
                    #acc_current[outpos].score += acc_prev[inpos].score
                    #inpos += 1
                    if len(acc_current) >= inpos + 1:
                        acc_current[outpos].doc_id = doc_id
                        acc_current[outpos].score = acc_prev[inpos].score + (math.log2(num_of_docs / query_term_freq[query_term]) \
                                                    * TFBM25(dict_doc_length[doc_id], pos_obj.termfrequency, doc_avg_length))
                    else:
                        acc_current.append(
                            Ranking(doc_id, acc_prev[inpos].score + (math.log2(num_of_docs / query_term_freq[query_term])
                                    * TFBM25(dict_doc_length[doc_id], pos_obj.termfrequency, doc_avg_length))))
                    #outpos += 1
                    inpos += 1
                else:
                    acc_current.append(Ranking(doc_id, (math.log2(num_of_docs / query_term_freq[query_term]) *
                                               TFBM25(dict_doc_length[doc_id],pos_obj.termfrequency,doc_avg_length))))
                pos_obj = nextDoc(query_term,doc_id)
                if pos_obj == float("inf"):
                    doc_id = float("inf")
                else:
                    doc_id = pos_obj.doc_index
                outpos += 1
            # copy the remaining doc scores from previous terms
            while acc_prev[inpos].doc_id < float("inf"):
                acc_current.append(acc_prev[inpos])
                outpos += 1
                inpos += 1
            if acc_current[len(acc_current)-1].doc_id != float("inf"):
                acc_current.append(Ranking(float("inf"),0))
            # Swap the acc_prev and acc_current
            temp_lst = acc_prev
            acc_prev = acc_current
            acc_current = temp_lst


        # when no quota is left
        elif quota_left == 0:
            for i in range(0,len(acc_prev)-1):
                pos_obj= nextDoc(query_term,acc_prev[i].doc_id-1)
                if pos_obj != float("inf") and pos_obj.doc_index == acc_prev[i].doc_id:
                    acc_prev[i].score += (math.log2(num_of_docs/query_term_freq[query_term]) * \
                                        TFBM25(dict_doc_length[acc_prev[i].doc_id],pos_obj.termfrequency,doc_avg_length))

        # the entire posting will not fit within the available quota. But some quota is still left. Accumulator pruning
        # needs to be done to add the documents for scoring
        else:
            tfstats = {}
            threshold_val = 1
            postings_seen = 0
            inpos = 0
            outpos = 0

            pos_obj = nextDoc(query_term,float("-inf"))
            doc_id = pos_obj.doc_index
            while doc_id < float("inf"):
                while acc_prev[inpos].doc_id < doc_id:
                    if len(acc_current) >= inpos + 1:
                        acc_current[outpos] = acc_prev[inpos]
                    else:
                        acc_current.append(acc_prev[inpos])
                    outpos += 1
                    inpos += 1
                if acc_prev[inpos].doc_id == doc_id:
                    if len(acc_current) >= inpos + 1:
                        acc_current[outpos].doc_id = doc_id
                        acc_current[outpos].score = acc_prev[inpos].score + (math.log2(num_of_docs/query_term_freq[query_term])\
                                                * TFBM25(dict_doc_length[doc_id],pos_obj.termfrequency,doc_avg_length))
                    else:
                        acc_current.append(Ranking(doc_id,acc_prev[inpos].score + (math.log2(num_of_docs/query_term_freq[query_term])
                                                * TFBM25(dict_doc_length[doc_id],pos_obj.termfrequency,doc_avg_length))))
                    outpos += 1
                    inpos += 1
                elif quota_left > 0:
                    if pos_obj.termfrequency > threshold_val:
                        acc_current.append(Ranking(doc_id,(math.log2(num_of_docs/query_term_freq[query_term])
                                                * TFBM25(dict_doc_length[doc_id],pos_obj.termfrequency,doc_avg_length))))
                        outpos += 1
                        quota_left -= 1
                    if pos_obj.termfrequency in tfstats:
                        tfstats[pos_obj.termfrequency] = tfstats[pos_obj.termfrequency] + 1
                    else:
                        tfstats[pos_obj.termfrequency] = 1
                postings_seen += 1
                if postings_seen % 2 == 0:
                    q = (query_term_freq[query_term] - postings_seen)/postings_seen

                    x_vals = list(tfstats.keys())
                    x_vals.sort()
                    quota_score = 0

                    for x in x_vals:
                        quota_score += tfstats[x] * q
                        if quota_score >= quota_left:
                            threshold_val = x
                            break

                pos_obj = nextDoc(query_term,doc_id)
                if pos_obj == float("inf"):
                    doc_id = float("inf")
                else:
                    doc_id = pos_obj.doc_index

            while acc_prev[inpos].doc_id < float("inf"):
                acc_current.append(acc_prev[inpos])
                outpos += 1
                inpos += 1
            if acc_current[len(acc_current)-1].doc_id != float("inf"):
                acc_current.append(Ranking(float("inf"),0))

            temp_lst = acc_prev
            acc_prev = acc_current
            acc_current = temp_lst
    acc_prev.pop()
    # sort the results according to the scores
    acc_prev.sort(key= lambda x: x.score,reverse=True)
    results = acc_prev[:int(num_of_results)]
    # write the output to trec_top_file
    writeFileContents("trec_top_file",results)
    #print("query_Id\titeration\tdoc_id\tranking\tscore\ttest_name")
    for result in results:
        print('0',' ','0',' ',result.doc_id,' ',results.index(result),' ', result.score,' ','my_test')



"""
Finds minimum of given list of (doc, positional lists) pairs.
if end of corpus (inf, inf)
"""


def min_current(currents):
    min_doc = Current("inf", "inf")
    for cur in currents:
        if min_doc.doc_id == "inf":
            min_doc = cur
        if min_doc.doc_id > cur.doc_id:
            min_doc = cur
        elif (min_doc.doc_id == cur.doc_id) and (min_doc.pos > cur.pos):
            min_doc = cur
    return min_doc


"""
Finds maximum of given list of (doc, positional lists) pairs.
if end of corpus (inf, inf)
"""


def max_current(currents):
    max_doc = Current(-1, -1)
    for cur in currents:
        if cur.doc_id == "inf":
            return Current("inf", "inf")
        if max_doc.doc_id < cur.doc_id:
            max_doc = cur
        elif (max_doc.doc_id == cur.doc_id):
            if (max_doc.pos <= cur.pos):
                max_doc = cur
    return max_doc


"""
Finds next cover of query terms given corpus and position.
:param terms: query terms
:param position current position to start searching for cover
:param adt - has the inverted index adt 
"""


def nextCover(terms, position, adt):
    nextv = []
    nextu = []
    for term in terms:
        nextv.append(adt.next(term, position))
    v = max_current(nextv)
    if v.doc_id == "inf" or v.pos == "inf":
        return Cover("inf", "inf")
    for term in terms:
        # next word in the corpus after v; finds v+1
        v_1 = nextCorpusWord(v)
        nextu.append(adt.prev(term, v_1))
    u = min_current(nextu)

    if u.doc_id == v.doc_id:
        myc = Cover(u, v)
        return myc
    else:
        return nextCover(terms, u, adt)


"""
Finds the ranking for each cover
"""


def rankProximity(terms, adt):
    result = []
    uv = nextCover(terms, Current("-inf", "-inf"), adt)
    if uv.u == "inf" and uv.v == "inf":
        print("No cover found")
        return result
    d = uv.u.doc_id
    score = 0
    while uv.u != "inf":
        if d < uv.u.doc_id:
            result.append(Ranking(d, score))
            d = uv.u.doc_id
            score = 0
        score = score + 1 / uv.dist(1)
        uv = nextCover(terms, uv.u, adt)
    if d != "inf":
        result.append(Ranking(d, score))
    # sort results
    result.sort(key=lambda x: x.score, reverse=True)
    return result


def nextCorpusWord(v):
    v_len = corpus_doc_freq[v.doc_id]
    if v.pos + 1 < v_len:
        # one more term present in same doc
        return Current(v.doc_id, v.pos + 1)
    elif v.pos + 1 == v_len:
        # not end of corpus but end of current document, return first pos in next doc
        if (len(corpus_doc_freq) - 2) > v.doc_id:
            return Current(v.doc_id + 1, 0)
        else:
            # end of corpus
            return Current("inf", "inf")
    else:
        # invalid position
        return Current("inf", "inf")


def proximityRanking(corpus, query, num_of_results):
    (num_of_docs, term_dict) = createPositionalIndex(corpus)
    adt = InvertedIndex(term_dict, num_of_docs)
    query = Removepunctuation(query)
    rankings = rankProximity(query.lower().split(), adt)
    truncated = rankings[:int(num_of_results)]
    writeFileContents("trec_top_file", truncated)
        # print("DocId\tScore")
    for result in truncated:
        print('0', ' ', '0', ' ', result.doc_id, ' ', truncated.index(result), ' ', result.score, ' ', 'my_test')
    #for r in truncated:
     #   print(r.doc_id, "\t", r.score)


term_dict = {}
query_dict = {}
filename = sys.argv[1]
filecontents = getFileContents(filename)
ranking = sys.argv[2]
num_of_results = sys.argv[3]
query = sys.argv[4]



# To store the corpus length for each document
corpus_doc_freq = {-1: -1}

if ranking == 'proximity':
    proximityRanking(filecontents, query, num_of_results)

if ranking == 'bm25':
    BM25(filecontents,query,num_of_results)



