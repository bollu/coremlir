{-# LANGUAGE CPP #-}

module Core2MLIR.Plugin where

import Data.Maybe
import qualified Data.ByteString.Lazy as BSL
import qualified Codec.Serialise as Ser
import GhcPlugins hiding (TB)
#if !MIN_VERSION_ghc(8,8,0)
import CoreMonad (pprPassDetails)
#endif
import ErrUtils (showPass)
import GHC
import Pretty
import CorePrep
import Text.Printf
import System.FilePath
import System.Directory
import System.IO (withFile, IOMode(..))

import Core2MLIR.Convert
import Core2MLIR.ConvertToMLIR

plugin :: Plugin
plugin = defaultPlugin { installCoreToDos = install }

-- where phase' = showSDocDump dflags (ppr todo GhcPlugins.<> text ":" <+> pprPassDetails todo)

install :: [CommandLineOption] -> [CoreToDo] -> CoreM [CoreToDo]
install _opts todo = do
    dflags <- getDynFlags
    hscenv <- getHscEnv
    return [CoreDoPluginPass "dumpIntersperse" (liftIO . dumpIntersperse dflags 0 "dumpIntersperse: Dump BeforeCorePrep"),
            CoreDoPluginPass "genMLIR" (liftIO . genMLIR dflags 0 "Core2MLIR: GenMLIR BeforeCorePrep"),
            CoreDoPluginPass "RunCorePrep" (liftIO . runCorePrep dflags hscenv),
            CoreDoPluginPass "dumpIntersperse" (liftIO . dumpIntersperse dflags 1 "dumpIntersperse: Dump AfterCorePrep"),
            CoreDoPluginPass "genMLIR" (liftIO . genMLIR dflags 1 "Core2MLIR: GenMLIR AfterCorePrep")]


runCorePrep :: DynFlags -> HscEnv -> ModGuts -> IO ModGuts
runCorePrep dflags hscenv guts = do
    putStrLn $ "running CorePrep..."
    -- let hscenv =  _ :: HscEnv
    -- hscenv <- liftIO $ getHscEnv
    -- core_prep_env <- mkInitialCorePrepEnv hscenv
    let module_ = mg_module guts
    -- | https://haskell-code-explorer.mfix.io/package/ghc-8.6.1/show/main/HscMain.hs#L1590
    -- Synthesize a location from thin air
    let modloc =  ModLocation{ ml_hs_file = Nothing, 
                               ml_hi_file = panic "CorePrepSynthLoc:ml_hi_file",
                               ml_obj_file  = panic "CorePrepSynthLoc:ml_obj_file" }
    let coreprogram = mg_binds guts
    let tcs = mg_tcs guts
    (coreprogram', _) <- corePrepPgm hscenv module_ modloc coreprogram tcs
    let guts' = guts { mg_binds = coreprogram' }
    putStrLn $ "ran CorePrep..."
    return guts'


    -- let prefix = fromMaybe "dump" $ dumpPrefix dflags
    --     fname = printf "%spass-%04u.cbor" prefix n
    -- showPass dflags $ "GhcDump: Dumping core to "++fname
    -- let in_dump_dir = maybe id (</>) (dumpDir dflags)
    -- createDirectoryIfMissing True $ takeDirectory $ in_dump_dir fname
    -- BSL.writeFile (in_dump_dir fname) $ Ser.serialise (cvtModule phase guts)
    -- return guts

dumpIntersperse :: DynFlags -> Int -> String -> ModGuts -> IO ModGuts
dumpIntersperse dflags n phase guts = do
    let prefix = fromMaybe "dump" $ dumpPrefix dflags
        fname = printf "%spass-%04u.cbor" prefix n
    showPass dflags $ "GhcDump: Dumping core to "++fname
    let in_dump_dir = maybe id (</>) (dumpDir dflags)
    createDirectoryIfMissing True $ takeDirectory $ in_dump_dir fname
    BSL.writeFile (in_dump_dir fname) $ Ser.serialise (cvtModule phase guts)
    return guts


genMLIR :: DynFlags -> Int -> String -> ModGuts -> IO ModGuts
genMLIR dflags n phase guts = do
    let prefix = fromMaybe "dump" $ dumpPrefix dflags
    let fname = printf "tomlir-%spass-%04u.mlir" prefix n

    putStrLn $ "running Core2MLIR"
    showPass dflags $ "GhcDump: Dumping MLIR to "++ fname
    let in_dump_dir = maybe id (</>) (dumpDir dflags)
    createDirectoryIfMissing True $ takeDirectory $ in_dump_dir fname
    withFile (in_dump_dir fname) WriteMode (\h -> printSDoc PageMode dflags h (defaultDumpStyle dflags) (cvtModuleToMLIR dflags phase guts))
    putStrLn $ "ran core2MLIR"
    return guts
