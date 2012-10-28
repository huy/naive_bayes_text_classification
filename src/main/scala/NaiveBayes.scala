import scala.collection.mutable.{HashMap,HashSet}

/**
Implements Multinominal Naive Bayes text classiﬁcation algorithm from Text Book
"Introduction to Information Retrieval" By Christopher D. Manning, Prabhakar Raghavan & Hinrich Schütze
**/
class NaiveBayes[C](debug: Boolean = false) {
   case class KlassInfo(var nDocs: Int, var nTerms: Int, var termFreq: HashMap[String,Int])

   var allKlassInfo = new HashMap[C,KlassInfo]
   var vocabulary = new HashMap[String,Int]
   var nTerms = 0
   var nDocs = 0

   def train(klass: C, doc: Iterable[String]) = {
     if( !allKlassInfo.contains(klass) ){
       allKlassInfo += (klass->KlassInfo(0,0,new HashMap[String,Int]))
     }

     val klassInfo = allKlassInfo(klass)
     klassInfo.nDocs += 1
     nDocs += 1
     doc.foreach { term=>
       if( !klassInfo.termFreq.contains(term) )
         klassInfo.termFreq += (term->1)
       else
         klassInfo.termFreq(term) += 1
       klassInfo.nTerms += 1

       if( !vocabulary.contains(term) )
         vocabulary += (term->1)
       else
         vocabulary(term) += 1
       nTerms += 1
     }
   }

   def apply(docId: String, doc: Iterable[String]): Option[C] = {

     if(doc.forall{ t=> !vocabulary.contains(t) }){
       if(debug)
         println("--vocabulary contains no terms from  %s".format(docId))

       return None
     }

     val scorePerKlass = allKlassInfo.keys.map{ klass=> (klass,score(klass, doc)) }

     if(debug)
       println("--scorePerKlass against %s:\n%s".format(docId, scorePerKlass.mkString(", ")))

     if(scorePerKlass.groupBy{ case (klass, score) => score }.size == 1){
        if(debug)
          println("--no discrimination for %s".format(docId))
        None
       }
     else
       Some(scorePerKlass.maxBy{_._2}._1)
   }

   private def probabilityTermGivenKlass(term: String, klass: C): Double={
     val klassInfo = allKlassInfo(klass)
     val freq = klassInfo.termFreq.getOrElse(term,0)

     if(freq == 0) 1.0/vocabulary.size else (freq + 0.0)/klassInfo.nTerms
   }

   private def probabilityTerm(term: String): Double={
     val freq = vocabulary.getOrElse(term,0)

     if(freq == 0) 1.0/vocabulary.size else (freq + 0.0)/nTerms
   }

   private def score(klass: C, doc: Iterable[String]): Double = {
     val klassInfo = allKlassInfo(klass)
     val probabilityDocGivenKlass = (klassInfo.nDocs + 0.0)/nDocs

     val numer = doc.foldLeft(probabilityDocGivenKlass){(product,t) => 
       product*probabilityTermGivenKlass(t,klass)
     }
     val denom = doc.foldLeft(1.0){(product,t) =>
       product*probabilityTerm(t)
     }

     numer/denom
   }
}
