import scala.collection.mutable.{HashMap,HashSet}

/**
Implements Multinominal Naive Bayes text classiﬁcation algorithm from Text Book
"Introduction to Information Retrieval" By Christopher D. Manning, Prabhakar Raghavan & Hinrich Schütze
**/
class NaiveBayes[C](debug: Boolean = false) {
   case class KlassInfo(var nDocs: Int, var nTerms: Int, var termFreq: HashMap[String,Int])

   var allKlassInfo = new HashMap[C,KlassInfo]
   var vocabulary = new HashSet[String]
   var nDocs = 0

   def train(klass: C, doc: Iterable[String]) = {
     if (!allKlassInfo.contains(klass)) {
       allKlassInfo += (klass -> KlassInfo(0,0,new HashMap[String,Int]))
     }

     val klassInfo = allKlassInfo(klass)
     klassInfo.nDocs += 1
     nDocs += 1
     doc.foreach { term =>
       if (!klassInfo.termFreq.contains(term))
         klassInfo.termFreq += (term -> 1)
       else
         klassInfo.termFreq(term) += 1
       klassInfo.nTerms += 1

       vocabulary += term
     }
   }

   def apply(docId: String, doc: Iterable[String]): Option[C] = {

     val scorePerKlass = allKlassInfo.keys.map {klass => (klass,score(klass, doc))}

     if (debug)
       println("--scorePerKlass against %s:\n%s".format(docId, scorePerKlass.mkString(", ")))

     if (scorePerKlass.groupBy {case (klass, score) => score}.size == 1) {
       if(debug)
         println("--no discrimination for %s".format(docId))
       None
     }
     else
       Some(scorePerKlass.maxBy{_._2}._1)
   }

   private def probabilityTermGivenKlass(term: String, klass: C): Double = {
     val klassInfo = allKlassInfo(klass)
     val freq = klassInfo.termFreq.getOrElse(term,0)
     // Laplace add 1 smoothing
     (freq + 1.0)/(klassInfo.nTerms + vocabulary.size)
   }

   private def score(klass: C, doc: Iterable[String]): Double = {
     val klassInfo = allKlassInfo(klass)
     val probabilityKlass = (klassInfo.nDocs + 0.0)/nDocs

     doc.foldLeft(math.log(probabilityKlass)) { (sum,term) => 
       sum + math.log(probabilityTermGivenKlass(term, klass))
     }
   }
}
