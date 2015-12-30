import org.scalatest.FunSuite
 
class NaiveBayesTest extends FunSuite {
 
  test("class info") {
     val nb = new NaiveBayes[String]
     
     nb.train("china","Chinese Beijing Chinese".split(" "))
     nb.train("other","Tokyo Japan Chinese".split(" "))

     assertResult(4) {nb.vocabulary.size}

     assertResult(2) {nb.nDocs}    

     assertResult(3) {nb.allKlassInfo("china").nTerms}

     assertResult(1) {nb.allKlassInfo("china").nDocs}

     assertResult(2) {nb.allKlassInfo("china").termFreq("Chinese")}
  }

  test("example 13.1") {
     val nb = new NaiveBayes[String](debug = true)

     Map("Chinese Beijing Chinese"->"china",
         "Chinese Chinese Shanghai"->"china",
         "Chinese Macao"->"china",
         "Tokyo Japan Chinese"->"other").foreach{ case (str,country)=>
       nb.train(country,str.split(" "))
     }

     assertResult(Some("china")) {nb.apply("1", "Chinese Chinese Chinese Tokyo Japan".split(" "))}
     assertResult(Some("other")) {nb.apply("2", "Tokyo Japan".split(" "))}
  }
 
}
